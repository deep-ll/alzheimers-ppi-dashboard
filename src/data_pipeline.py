# Data Extraction & Graph Construction
"""
src/data_pipeline.py
Downloads the human PPI network AND sequences from STRING DB, 
samples 100,000 edges, generates REAL ProtBERT embeddings, 
and builds a PyTorch Geometric Data object.
"""
import os
import torch
import pandas as pd
import requests
import gzip
from tqdm import tqdm
from torch_geometric.data import Data

# Import your extractor (assuming this is run from the project root)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extractor import ProteinFeatureExtractor

def download_file(url, file_path):
    """Helper to download files in chunks."""
    if not os.path.exists(file_path):
        print(f"Downloading {url.split('/')[-1]}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"File {file_path} already exists. Skipping download.")
    return file_path

def load_fasta_sequences(fasta_path, target_proteins):
    """Extracts true amino acid sequences only for our selected proteins."""
    print("Parsing STRING DB FASTA file for target sequences...")
    sequences = {}
    current_protein = ""
    current_seq = []
    
    with gzip.open(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_protein in target_proteins:
                    sequences[current_protein] = "".join(current_seq)
                # The STRING fasta header looks like ">9606.ENSP00000000233"
                current_protein = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
                
        # Catch the last sequence in the file
        if current_protein in target_proteins:
            sequences[current_protein] = "".join(current_seq)
            
    return sequences

def build_graph(raw_dir, processed_dir, num_edges=100000):
    # 1. Download Links and Sequences
    links_url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    fasta_url = "https://stringdb-downloads.org/download/protein.sequences.v12.0/9606.protein.sequences.v12.0.fa.gz"
    
    links_path = download_file(links_url, os.path.join(raw_dir, "9606.links.txt.gz"))
    fasta_path = download_file(fasta_url, os.path.join(raw_dir, "9606.sequences.fa.gz"))

    # 2. Parse edges and sample
    print("Loading network data...")
    df = pd.read_csv(links_path, sep=' ')
    df_high_conf = df[df['combined_score'] >= 700].copy()
    df_sampled = df_high_conf.sample(n=num_edges, random_state=42).copy() if len(df_high_conf) > num_edges else df_high_conf.copy()
    
    # 3. Get unique proteins and map to IDs
    unique_proteins = set(df_sampled['protein1']).union(set(df_sampled['protein2']))
    protein_to_idx = {protein: idx for idx, protein in enumerate(unique_proteins)}
    
    df_sampled['src'] = df_sampled['protein1'].map(protein_to_idx)
    df_sampled['dst'] = df_sampled['protein2'].map(protein_to_idx)
    edge_index = torch.tensor([df_sampled['src'].values, df_sampled['dst'].values], dtype=torch.long)
    
    # 4. Extract Real Sequences
    protein_sequences = load_fasta_sequences(fasta_path, unique_proteins)
    
    # 5. Generate Real ProtBERT Embeddings
    print("\nInitializing ProtBERT (This will take a moment)...")
    extractor = ProteinFeatureExtractor()
    
    num_nodes = len(unique_proteins)
    feature_dim = 1024
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    
    print(f"Generating embeddings for {num_nodes} unique proteins...")
    # We sort by index so the tensor perfectly matches the edge_index mapping
    for protein, idx in tqdm(sorted(protein_to_idx.items(), key=lambda item: item[1])):
        raw_sequence = protein_sequences.get(protein, "")
        
        if raw_sequence:
            # Pass the raw amino acid string directly to your existing extractor
            # It returns a [1, 1024] tensor, which we squeeze and assign to the node row
            embedding = extractor.get_embedding(raw_sequence)
            x[idx] = embedding.squeeze(0).cpu() 
        else:
            # Fallback for missing sequences (very rare in STRING)
            print(f"Warning: No sequence found for {protein}")
            x[idx] = torch.zeros(feature_dim)
            
    # 6. Build and Save PyG Object
    data = Data(x=x, edge_index=edge_index)
    print("\n✅ Final PyG Graph with REAL ProtBERT features created:")
    print(data)
    
    # Save everything securely to Google Drive
    pd.DataFrame(list(protein_to_idx.items()), columns=['string_protein_id', 'node_idx']).to_csv(
        os.path.join(processed_dir, "node_mapping.csv"), index=False
    )
    torch.save(data, os.path.join(processed_dir, "pyg_ppi_graph.pt"))
    print("✅ All real data saved to Google Drive.")

if __name__ == "__main__":
    DRIVE_BASE = "/content/drive/MyDrive/Alzheimers_PPI_Project"
    RAW_DIR = os.path.join(DRIVE_BASE, "raw_data")
    PROCESSED_DIR = os.path.join(DRIVE_BASE, "processed")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    build_graph(RAW_DIR, PROCESSED_DIR, num_edges=100000)
