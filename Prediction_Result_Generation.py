#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pickle
import numpy as np
import torch
from torch import nn
from torch.utils import data
import os
from collections import defaultdict
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TEST_PATH ="dataset/TE197.txt"
EMBEDDINGS_BASE_PATH = "embedding/test197_embeddings.pkl"

MODEL_CONFIG = {
    "window_size": 30,
    "feature_dim": 1024,
    "hidden_dim": 32,
    "ssd_scales": [0.17, 0.5, 1.0]
}

def load_test_data():
    proteins_list = []
    
    with open(EMBEDDINGS_BASE_PATH, "rb") as f:
        embeddings = pickle.load(f)
        
    with open(TEST_PATH, "r") as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        pro_id = lines[i].strip()[1:]
        pro_seq = lines[i + 1].strip()
        pro_label = [int(x) for x in lines[i + 2].strip()]
        
        if pro_id not in embeddings:
            print(f"Warning: ID {pro_id} not found in embeddings")
            i += 3
            continue
            
        proteins_list.append({
            "id": pro_id,
            "seq": pro_seq,
            "embeddings": embeddings[pro_id],
            "label": pro_label
        })
        i += 3
        
    print(f"Loaded test proteins: {len(proteins_list)}")
    return proteins_list

def generate_samples(proteins, window_size):
    samples = []
    for pro in proteins:
        pro_id = pro["id"]
        pro_embeddings = pro["embeddings"]
        pro_seq = pro["seq"]
        aa_idx = 0
        
        if len(pro_embeddings) != len(pro_seq):
            print(f"Warning: ID {pro_id} embedding length mismatch")
            continue
            
        while aa_idx < len(pro_seq):
            sample_seq = pro_seq[aa_idx:aa_idx + window_size]
            sample_embeddings = pro_embeddings[aa_idx:aa_idx + window_size]
            
            if len(sample_seq) < window_size:
                padding_length = window_size - len(sample_seq)
                sample_seq += "-" * padding_length
                sample_embeddings = np.concatenate(
                    (sample_embeddings, np.zeros((padding_length, sample_embeddings.shape[1]))))
                
            sample_mask = [1 if x != "-" else 0 for x in sample_seq]
            samples.append((pro_id, aa_idx, sample_seq, sample_embeddings, sample_mask))
            aa_idx += window_size
            
    return samples

class SiteDataset(data.Dataset):
    def __init__(self, samples):
        super(SiteDataset, self).__init__()
        self.samples = samples

    def __getitem__(self, index):
        pro_id, aa_idx, _, embeddings, mask = self.samples[index]
        feature = torch.tensor(embeddings, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return pro_id, aa_idx, feature, mask

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
            nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1),
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

class ReduceDimLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReduceDimLayer, self).__init__()
        self.conv_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
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
        self.attention_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=num_heads)
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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, X):
        return self.dense(X).squeeze(-1)

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
            anchor_scales=MODEL_CONFIG["ssd_scales"]
        )
        
    def forward(self, X):
        reduced_X = self.reduce_dim_layer(X)
        encoded_X = self.encoder(reduced_X)
        site_logits = self.site_decoder(encoded_X)
        site_prob = self.sigmoid(site_logits)
        loc_pred, cls_pred, anchors = self.ssd_module(encoded_X)
        return site_prob, (loc_pred, cls_pred, anchors)

def predict_test_set(model, test_iter):
    model.eval()
    results_dict = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(test_iter, desc="Predicting test set"):
            pro_ids = batch[0]
            aa_idxs = batch[1]
            features = batch[2].to(device)
            masks = batch[3]
            site_prob, _ = model(features)
            site_prob = site_prob.cpu().numpy()
            masks = masks.numpy()
            for i in range(len(pro_ids)):
                pro_id = pro_ids[i]
                aa_idx = aa_idxs[i]
                preds = site_prob[i]
                mask = masks[i]
                valid_preds = preds[mask == 1]
                results_dict[pro_id].append((aa_idx, valid_preds))
    return results_dict

def combine_predictions(results_dict, proteins):
    full_predictions = {}
    for pro in proteins:
        pro_id = pro["id"]
        seq_len = len(pro["seq"])
        scores = np.zeros(seq_len, dtype=np.float32)
        counts = np.zeros(seq_len, dtype=np.int32)
        for aa_idx, window_preds in results_dict[pro_id]:
            end_idx = aa_idx + len(window_preds)
            scores[aa_idx:end_idx] += window_preds
            counts[aa_idx:end_idx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_scores = np.where(counts > 0, scores / counts, 0.0)
        full_predictions[pro_id] = {
            "sequence": pro["seq"],
            "scores": avg_scores
        }
    return full_predictions

def save_to_fasta(full_predictions, output_file):
    with open(output_file, "w") as f:
        for pro_id, data in full_predictions.items():
            sequence = data["sequence"]
            scores = " ".join([f"{score:.6f}" for score in data["scores"]])
            f.write(f">{pro_id}\n")
            f.write(f"{sequence}\n")
            f.write(f"{scores}\n")
    print(f"Predictions saved to: {output_file}")

def main():
    print("Loading test data...")
    test_proteins = load_test_data()
    print("Generating test samples...")
    window_size = MODEL_CONFIG["window_size"]
    test_samples = generate_samples(test_proteins, window_size)
    print(f"Generated samples: {len(test_samples)}")
    test_dataset = SiteDataset(test_samples)
    test_iter = data.DataLoader(
        test_dataset, 
        batch_size=64,
        shuffle=False,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch],
            torch.stack([item[2] for item in batch]),
            torch.stack([item[3] for item in batch]),
        )
    )
    print("Loading pre-trained model...")
    model_path = "best_model/best_model.pth"
    model = SitePredictor(
        input_dim=MODEL_CONFIG["feature_dim"],
        window_size=MODEL_CONFIG["window_size"],
        hidden_dim=MODEL_CONFIG["hidden_dim"]
    ).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded: {model_path}")
    print("Starting prediction...")
    results_dict = predict_test_set(model, test_iter)
    full_predictions = combine_predictions(results_dict, test_proteins)
    output_file = "prediction/test_predictions.fasta"
    save_to_fasta(full_predictions, output_file)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




