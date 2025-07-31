#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils import data
import time
import os
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split
import random

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TRAIN_PATH = "dataset/TR1000.fasta"
TEST440_PATH = "dataset/TE197.txt"
EMBEDDINGS_BASE_PATH = "embedding/{}_embeddings.pkl"

original_dataset_path = {
    "train": TRAIN_PATH,
    "test440": TEST440_PATH,
}

def load_all_data():
    proteins_dict = {}
    for key, path in original_dataset_path.items():
        proteins_dict[key] = []
        embedding_path = EMBEDDINGS_BASE_PATH.format(key)
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Protein file not found: {path}")
        
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)
            
        with open(path, "r") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            pro_id = lines[i].strip()[1:]
            pro_seq = lines[i + 1].strip()
            pro_label = [int(x) for x in lines[i + 2].strip()]
            
            if pro_id not in embeddings:
                print(f"Warning: ID {pro_id} missing in embeddings")
                i += 3
                continue
                
            proteins_dict[key].append({
                "id": pro_id,
                "seq": pro_seq,
                "embeddings": embeddings[pro_id],
                "label": pro_label
            })
            i += 3
            
    return proteins_dict

def generate_samples(proteins, window_size):
    samples = []
    for pro in proteins:
        pro_id = pro["id"]
        pro_embeddings = pro["embeddings"]
        pro_seq = pro["seq"]
        pro_label = pro["label"]
        aa_idx = 0
            
        if len(pro_embeddings) != len(pro_seq):
            print(f"Warning: ID {pro_id} has embedding length ({len(pro_embeddings)}) that doesn't match sequence length ({len(pro_seq)})")
            continue
            
        while aa_idx < len(pro_seq):
            sample_seq = pro_seq[aa_idx:aa_idx + window_size]
            sample_label = pro_label[aa_idx:aa_idx + window_size]
            sample_embeddings = pro_embeddings[aa_idx:aa_idx + window_size]
            
            if len(sample_seq) < window_size:
                padding_length = window_size - len(sample_seq)
                sample_seq += "-" * padding_length
                sample_label += [0] * padding_length
                sample_embeddings = np.concatenate(
                    (sample_embeddings, np.zeros((padding_length, sample_embeddings.shape[1]))))
                
            sample_mask = [1 if x != "-" else 0 for x in sample_seq]
            samples.append((pro_id, aa_idx, sample_seq, sample_embeddings, sample_label, sample_mask))
            aa_idx += window_size
            
    return samples

def calculate_sequence_stats(proteins):
    total_residues = 0
    positive_residues = 0
    for pro in proteins:
        total_residues += len(pro['seq'])
        positive_residues += sum(pro['label'])
    
    ratio = positive_residues / total_residues if total_residues > 0 else 0
    return total_residues, positive_residues, ratio

class SiteDataset(data.Dataset):
    def __init__(self, samples):
        super(SiteDataset, self).__init__()
        self.samples = samples

    def __getitem__(self, index):
        _, _, _, embeddings, label, mask = self.samples[index]
        feature = torch.tensor(embeddings, dtype=torch.float32)
        site_label = torch.tensor(label, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return feature, site_label, mask

    def __len__(self):
        return len(self.samples)

class SSDPredictor(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(SSDPredictor, self).__init__()
        self.loc_pred = nn.Conv1d(in_channels, num_anchors * 1, kernel_size=3, padding=1)
        self.cls_pred = nn.Conv1d(in_channels, num_anchors * 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        loc_pred = self.loc_pred(x)
        cls_pred = self.cls_pred(x)
        
        loc_pred = loc_pred.permute(0, 2, 1).contiguous().view(x.size(0), -1, 1)
        cls_pred = cls_pred.permute(0, 2, 1).contiguous().view(x.size(0), -1)
        
        return loc_pred, cls_pred

class SSDModule(nn.Module):
    def __init__(self, feature_dim, window_size, anchor_scales=[0.17, 0.5, 1.0]):
        super(SSDModule, self).__init__()
        self.window_size = window_size
        self.anchor_scales = anchor_scales
        self.num_anchors = len(anchor_scales)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.predictor = SSDPredictor(32, self.num_anchors)
        
        self.register_buffer('anchors', self.generate_anchors())
        
    def generate_anchors(self):
        anchors = []
        for pos in range(self.window_size):
            for scale in self.anchor_scales:
                center = (pos + 0.5) / self.window_size
                span = int(scale * self.window_size)
                anchors.append([center, span])
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.feature_extractor(x)
        loc_pred, cls_pred = self.predictor(features)
        return loc_pred, cls_pred, self.anchors

class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        pos_mask = (cls_targets > 0.5).float()
        num_pos = torch.sum(pos_mask).item()
        
        if num_pos == 0:
            loc_loss = torch.tensor(0.0, device=loc_preds.device)
            
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds, 
                cls_targets.float(),
                reduction='mean'
            )
            return loc_loss + cls_loss
        
        loc_loss = F.smooth_l1_loss(
            loc_preds * pos_mask.unsqueeze(-1), 
            loc_targets * pos_mask.unsqueeze(-1), 
            reduction='sum'
        ) / num_pos
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_preds, 
            cls_targets.float(),
            reduction='none'
        )
        
        with torch.no_grad():
            loss = -cls_loss
            loss[pos_mask.bool()] = -float('inf')
            
            _, loss_idx = loss.sort(dim=1, descending=True)
            _, idx_rank = loss_idx.sort(dim=1)
            
            num_neg = min(int(self.neg_pos_ratio * num_pos), 
                         pos_mask.size(1) - num_pos)
            num_neg = max(0, num_neg)
            neg_mask = idx_rank < num_neg
        
        cls_weight = torch.ones_like(cls_loss)
        cls_weight[~neg_mask & ~pos_mask.bool()] = 0.0
            
        cls_loss = (cls_loss * cls_weight).sum() / num_pos
        
        return loc_loss + cls_loss

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score

def evaluate_sites(data_iter, model, device, return_preds=False):
    site_prob_all = []
    site_label_all = []
    mask_all = []
    total_loss = 0.0
    site_criterion = nn.BCELoss(reduction='sum')
    ssd_criterion = SSDLoss()
    total_residues = 0
    
    model.eval()
    with torch.no_grad():
        for features, site_labels, masks in data_iter:
            features = features.to(device)
            site_labels = site_labels.to(device)
            masks = masks.to(device)
            batch_size = features.size(0)
            
            site_prob, (loc_pred, cls_pred, anchors) = model(features)
            
            valid_mask = (masks == 1)
            site_loss = site_criterion(site_prob[valid_mask], site_labels[valid_mask])
            
            loc_targets, cls_targets = prepare_ssd_targets(site_labels, anchors, train_config["window_size"])
            ssd_loss = ssd_criterion(loc_pred, loc_targets, cls_pred, cls_targets)
            
            total_loss += (site_loss + ssd_loss).item()
            
            site_prob_all.append(site_prob.detach().cpu())
            site_label_all.append(site_labels.cpu())
            mask_all.append(masks.cpu())
            
            total_residues += valid_mask.sum().item()
            
        site_prob_all = torch.cat(site_prob_all, dim=0)
        site_label_all = torch.cat(site_label_all, dim=0)
        mask_all = torch.cat(mask_all, dim=0)
        
        mask_all = mask_all.flatten()
        site_prob_flat = site_prob_all.flatten()[mask_all == 1]
        site_label_flat = site_label_all.flatten()[mask_all == 1]
        
        if site_label_flat.size(0) == 0:
            print("Warning: No valid residues for evaluation")
            auc_score = 0.0
            aupr_score = 0.0
        else:
            auc_score = roc_auc_score(site_label_flat.numpy(), site_prob_flat.numpy())
            precision, recall, _ = precision_recall_curve(site_label_flat.numpy(), site_prob_flat.numpy())
            aupr_score = auc(recall, precision)
        
        avg_loss = total_loss / total_residues if total_residues > 0 else 0
        
        if return_preds:
            return auc_score, aupr_score, avg_loss, (site_prob_flat, site_label_flat)
        else:
            return auc_score, aupr_score, avg_loss

def prepare_ssd_targets(site_labels, anchors, window_size):
    batch_size = site_labels.size(0)
    num_anchors = anchors.size(0)
    
    loc_targets = torch.zeros(batch_size, num_anchors, 1, device=site_labels.device)
    cls_targets = torch.zeros(batch_size, num_anchors, device=site_labels.device)
    
    for i in range(batch_size):
        seq_labels = site_labels[i]
        pos_indices = (seq_labels > 0.5).nonzero(as_tuple=True)[0]
        
        for j, anchor in enumerate(anchors):
            center, span = anchor
            center_idx = int(center * window_size)
            anchor_start = max(0, center_idx - span // 2)
            anchor_end = min(window_size, center_idx + span // 2)
            
            for idx in pos_indices:
                if anchor_start <= idx.item() < anchor_end:
                    cls_targets[i, j] = 1.0
                    loc_target = (idx.float() / window_size) - center
                    loc_targets[i, j, 0] = loc_target
                    break
    
    return loc_targets, cls_targets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
try:
    proteins_dict = load_all_data()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Data loading failed: {str(e)}")
    raise

train_config = {
    "window_size": 30,
    "batch_size": 64,
    "lr": 0.003,
    "epochs": 2,  
    "feature_dim": 1024,
    "ssd_scales": [0.17, 0.5, 1.0],
    "val_size": 0.2
}
window_size = train_config["window_size"]

print("Preparing training dataset...")
train_data = proteins_dict["train"]

all_protein_ids = [pro["id"] for pro in train_data]
train_ids, val_ids = train_test_split(
    all_protein_ids, 
    test_size=train_config["val_size"], 
    random_state=50
)

train_proteins = [pro for pro in train_data if pro["id"] in train_ids]
val_proteins = [pro for pro in train_data if pro["id"] in val_ids]

train_total, train_positive, train_ratio = calculate_sequence_stats(train_proteins)
val_total, val_positive, val_ratio = calculate_sequence_stats(val_proteins)

print(f"Training proteins: {len(train_proteins)}")
print(f"Training residues: {train_total}, Positive residues: {train_positive}, Ratio: {train_ratio:.6f}")
print(f"Validation proteins: {len(val_proteins)}")
print(f"Validation residues: {val_total}, Positive residues: {val_positive}, Ratio: {val_ratio:.6f}")

print(f"Generating training samples ({len(train_proteins)} proteins)...")
train_samples = generate_samples(train_proteins, window_size)
print(f"Generating validation samples ({len(val_proteins)} proteins)...")
val_samples = generate_samples(val_proteins, window_size)

print(f"Training samples generated: {len(train_samples)}")
print(f"Validation samples generated: {len(val_samples)}")

train_dataset = SiteDataset(train_samples)
val_dataset = SiteDataset(val_samples)
train_iter = data.DataLoader(train_dataset, batch_size=train_config["batch_size"], 
                            shuffle=True, num_workers=4 if torch.cuda.is_available() else 2)
val_iter = data.DataLoader(val_dataset, batch_size=train_config["batch_size"], 
                          shuffle=False, num_workers=4 if torch.cuda.is_available() else 2)

print("Preparing test dataset...")
test440_data = proteins_dict["test440"]
test440_total, test440_positive, test440_ratio = calculate_sequence_stats(test440_data)
print(f"Test440 proteins: {len(test440_data)}")
print(f"Test440 residues: {test440_total}, Positive residues: {test440_positive}, Ratio: {test440_ratio:.6f}")

test440_samples = generate_samples(test440_data, window_size)
test440_dataset = SiteDataset(test440_samples)
test440_iter = data.DataLoader(test440_dataset, batch_size=train_config["batch_size"], 
                              shuffle=False, num_workers=4 if torch.cuda.is_available() else 2)
print(f"Test440 samples generated: {len(test440_samples)}")

class ReduceDimLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReduceDimLayer, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.permute(0, 2, 1)
        X = self.conv_layer(X)
        X = X.permute(0, 2, 1)
        X = self.relu(X)
        return X

class SiteEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, num_heads=4):
        super(SiteEncoder, self).__init__()
        self.attention_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.share_layer = nn.TransformerEncoder(self.attention_layer, num_layers=num_layers)

    def forward(self, X):
        X = X.permute(1, 0, 2)
        X = self.share_layer(X)
        X = X.permute(1, 0, 2)
        return X

class SiteDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiteDecoder, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim//2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim//2), 1)
        )

    def forward(self, X):
        X = self.dense(X).squeeze(-1)
        return X
        
class SitePredictor(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim=32):
        super(SitePredictor, self).__init__()
        self.reduce_dim_layer = ReduceDimLayer(input_dim, hidden_dim)
        self.encoder = SiteEncoder(hidden_dim)
        self.site_decoder = SiteDecoder(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.ssd_module = SSDModule(
            feature_dim=hidden_dim,
            window_size=window_size,
            anchor_scales=train_config["ssd_scales"]
        )
        
    def forward(self, X):
        reduced_X = self.reduce_dim_layer(X)
        encoded_X = self.encoder(reduced_X)
        
        site_logits = self.site_decoder(encoded_X)
        site_prob = self.sigmoid(site_logits)
        
        loc_pred, cls_pred, anchors = self.ssd_module(encoded_X)
        
        return site_prob, (loc_pred, cls_pred, anchors)

def main():
    print("Initializing model...")
    set_random_seed(seed=42)
    model = SitePredictor(
        input_dim=train_config["feature_dim"],
        window_size=train_config["window_size"],
        hidden_dim=32
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    site_criterion = nn.BCELoss()
    ssd_criterion = SSDLoss()
    
    best_val_auc = 0.0
    best_epoch = -1
    
    print("\nStarting training...")
    for epoch in range(train_config["epochs"]):
        model.train()
        train_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        for features, site_labels, masks in tqdm(train_iter, desc=f"Epoch {epoch+1}/{train_config['epochs']}"):
            features = features.to(device)
            site_labels = site_labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            site_prob, (loc_pred, cls_pred, anchors) = model(features)
            
            valid_mask = (masks == 1)
            site_loss = site_criterion(site_prob[valid_mask], site_labels[valid_mask])
            
            loc_targets, cls_targets = prepare_ssd_targets(site_labels, anchors, train_config["window_size"])
            
            ssd_loss = ssd_criterion(loc_pred, loc_targets, cls_pred, cls_targets)
            
            total_loss = site_loss + ssd_loss
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            total_samples += features.size(0)
        
        avg_train_loss = train_loss / len(train_iter)
        
        model.eval()
        
        print(f"\nEvaluating validation set...")
        val_auc, val_aupr, val_loss = evaluate_sites(val_iter, model, device)
        
        print(f"Evaluating test set...")
        test_auc, test_aupr, test_loss = evaluate_sites(test440_iter, model, device)
        
        print(f"Evaluating training set...")
        train_auc, train_aupr, _ = evaluate_sites(train_iter, model, device)
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{train_config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f}, AUC: {train_auc:.4f}, AUPR: {train_aupr:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")
        print(f"  Test  Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_aupr': val_aupr,
            }, "best_model/best_model.pth")
            print(f"  Best model saved! (Val AUC: {val_auc:.4f})")
    
    print(f"\nTraining complete! Best model at epoch {best_epoch+1}, Val AUC: {best_val_auc:.4f}")
    
    try:
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        checkpoint = torch.load("best_model/best_model.pth", weights_only=True)
    except:
        print("Warning: Secure loading failed, attempting non-secure load")
        checkpoint = torch.load("best_model/best_model.pth", weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_auc, test_aupr, test_loss, (test_probs, test_labels) = evaluate_sites(
        test440_iter, model, device, return_preds=True
    )
    
    if test_probs.numel() > 0 and test_labels.numel() > 0:
        test_preds = (test_probs > 0.5).float()
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds, zero_division=0)
        recall = recall_score(test_labels, test_preds, zero_division=0)
        f1 = f1_score(test_labels, test_preds, zero_division=0)
    else:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    print("\nFinal test results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  Test AUPR: {test_aupr:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

if __name__ == "__main__":
    main()

