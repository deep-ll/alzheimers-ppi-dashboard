import os
import torch
import pandas as pd
import h5py
import numpy as np
import requests
from torch_geometric.data import Data

def download_file(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {os.path.basename(file_path)}... (This might take a moment)")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"File {os.path.basename(file_path)} already exists. Skipping download.")
    return file_path

def build_graph(raw_dir, processed_dir, num_edges=100000):
    # 1. Download Links
    links_url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    links_path = download_file(links_url, os.path.join(raw_dir, "9606.links.txt.gz"))
    
    print("Loading edges directly from gzip...")
    df = pd.read_csv(links_path, sep=' ')
    
    # Filter for high confidence
    df_high_conf = df[df['combined_score'] >= 700].copy()
    
    if len(df_high_conf) > num_edges:
        df_sampled = df_high_conf.sample(n=num_edges, random_state=42).copy()
    else:
        df_sampled = df_high_conf.copy()
        
    unique_proteins = set(df_sampled['protein1']).union(set(df_sampled['protein2']))
    protein_to_idx = {protein: idx for idx, protein in enumerate(unique_proteins)}
    
    print(f"Graph contains {len(unique_proteins):,} unique proteins.")
    
    # 2. Build edge_index
    print("Building edge connections...")
    df_sampled['src'] = df_sampled['protein1'].map(protein_to_idx)
    df_sampled['dst'] = df_sampled['protein2'].map(protein_to_idx)
    
    edges_array = np.array([df_sampled['src'].values, df_sampled['dst'].values])
    edge_index = torch.tensor(edges_array, dtype=torch.long)
    
    # 3. Download and Load Pre-computed STRING Embeddings
    embeddings_url = "https://stringdb-downloads.org/download/protein.sequence.embeddings.v12.0/9606.protein.sequence.embeddings.v12.0.h5"
    h5_path = download_file(embeddings_url, os.path.join(raw_dir, "9606.protein.sequence.embeddings.v12.0.h5"))
    
    # Initialize all embeddings as zero first
    x = torch.zeros((len(unique_proteins), 1024), dtype=torch.float)
    found_count = 0
    
    print(f"Extracting embeddings from {os.path.basename(h5_path)}...")
    with h5py.File(h5_path, 'r') as h5_file:
        # The AI embeddings are stored in parallel arrays
        proteins_array = h5_file['proteins'][:]
        embeddings_matrix = h5_file['embeddings'][:]
        
        for i, p_bytes in enumerate(proteins_array):
            # Decode the bytes format STRING uses
            p_str = p_bytes.decode('utf-8') if isinstance(p_bytes, bytes) else str(p_bytes)
            
            # Robust mapping in case the '9606.' prefix differs
            if p_str in protein_to_idx:
                idx = protein_to_idx[p_str]
            elif f"9606.{p_str}" in protein_to_idx:
                idx = protein_to_idx[f"9606.{p_str}"]
            elif p_str.replace("9606.", "") in protein_to_idx:
                idx = protein_to_idx[p_str.replace("9606.", "")]
            else:
                continue
                
            # Assign the 1024-d vector to our PyTorch tensor
            x[idx] = torch.tensor(embeddings_matrix[i], dtype=torch.float)
            found_count += 1

    missing_proteins = len(unique_proteins) - found_count
    print(f"Embeddings loaded successfully! Found: {found_count:,} | Missing: {missing_proteins:,}")

    # 4. Save the graph object
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, os.path.join(processed_dir, "pyg_ppi_graph.pt"))
    
    # Save mapping
    mapping_df = pd.DataFrame(list(protein_to_idx.items()), columns=['string_protein_id', 'node_idx'])
    mapping_df.to_csv(os.path.join(processed_dir, "node_mapping.csv"), index=False)
    
    print(f"✅ Graph built and saved successfully to {processed_dir}")

if __name__ == "__main__":
    DRIVE_BASE = "/content/drive/MyDrive/Alzheimers_PPI_Project"
    RAW_DIR = os.path.join(DRIVE_BASE, "raw_data")
    PROCESSED_DIR = os.path.join(DRIVE_BASE, "processed")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    build_graph(RAW_DIR, PROCESSED_DIR, num_edges=100000)
