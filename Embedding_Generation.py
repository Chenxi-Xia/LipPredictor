#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

import os
import pickle
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
import gc
import sys

def load_protein_data(file_path):
    proteins = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        pro_id = lines[i].strip()[1:]
        pro_seq = lines[i+1].strip()
        pro_label = [int(x) for x in lines[i+2].strip()]
        proteins.append({
            "id": pro_id,
            "seq": pro_seq,
            "label": pro_label
        })
        i += 3
    
    return proteins

def generate_embeddings(proteins, tokenizer, model, device, batch_size=1, max_sequence_length=2000):
    embeddings = {}
    
    for i in tqdm(range(0, len(proteins), batch_size), desc="Generating embeddings"):
        batch_proteins = proteins[i:i+batch_size]
        sequences = []
        
        for protein in batch_proteins:
            seq = re.sub(r"[UZOB]", "X", protein["seq"])
            sequences.append(" ".join(list(seq)))
        
        token_encoding = tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="longest" if batch_size > 1 else False,
            return_tensors="pt"
        )
        
        input_ids = token_encoding["input_ids"].to(device)
        attention_mask = token_encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state.detach().cpu().numpy()
        
        for j, protein in enumerate(batch_proteins):
            seq_len = len(protein["seq"])
            protein_emb = last_hidden_states[j]
            
            if batch_size == 1:
                residue_embeddings = protein_emb[1:1+seq_len]
            else:
                actual_mask = attention_mask[j].detach().cpu().numpy()
                actual_len = np.sum(actual_mask) - 2
                residue_embeddings = protein_emb[1:1+actual_len][:seq_len]
            
            embeddings[protein["id"]] = residue_embeddings
    
    return embeddings

def save_embeddings(embeddings, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

def load_model(device):
    model_name = "prot_t5_xl_uniref50/prot_t5_xl_uniref50"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    print(f"Loading model: {model_name}")
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model = model.eval()
    
    return tokenizer, model

def main():
    dataset_path = "dataset/TR1000.fasta"
    output_path = "embedding/train_embeddings.pkl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading protein data...")
    proteins = load_protein_data(dataset_path)
    print(f"Loaded {len(proteins)} proteins")
    
    tokenizer, model = load_model(device)
    
    batch_size = 1
    max_sequence_length = 2000
    
    short_proteins = [p for p in proteins if len(p["seq"]) <= max_sequence_length]
    long_proteins = [p for p in proteins if len(p["seq"]) > max_sequence_length]
    
    print(f"Short sequence proteins (<={max_sequence_length}): {len(short_proteins)}")
    print(f"Long sequence proteins (>{max_sequence_length}): {len(long_proteins)}")
    
    all_embeddings = {}
    
    if short_proteins:
        embeddings_short = generate_embeddings(
            short_proteins, 
            tokenizer, 
            model, 
            device, 
            batch_size=batch_size,
            max_sequence_length=max_sequence_length
        )
        all_embeddings.update(embeddings_short)
    
    if long_proteins:
        print("Processing long sequences...")
        for protein in tqdm(long_proteins, desc="Long sequences"):
            try:
                seq = re.sub(r"[UZOB]", "X", protein["seq"])
                token_encoding = tokenizer.encode_plus(
                    " ".join(list(seq)),
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                input_ids = token_encoding["input_ids"].to(device)
                attention_mask = token_encoding["attention_mask"].to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_states = outputs.last_hidden_state.detach().cpu().numpy().squeeze()
                
                residue_embeddings = last_hidden_states[1:1+len(protein["seq"])]
                all_embeddings[protein["id"]] = residue_embeddings
                
                del input_ids, attention_mask, outputs, last_hidden_states
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Failed to process sequence {protein['id']} (length {len(protein['seq'])}): {str(e)}")
    
    save_embeddings(all_embeddings, output_path)
    
    print(f"Successfully generated embeddings for {len(all_embeddings)} proteins")
    if all_embeddings:
        sample_id = next(iter(all_embeddings.keys()))
        sample_emb = all_embeddings[sample_id]
        print(f"Sample embedding shape (id: {sample_id}): {sample_emb.shape}")

if __name__ == "__main__":
    main()

