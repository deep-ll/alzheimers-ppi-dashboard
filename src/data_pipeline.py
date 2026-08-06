# Data Extraction & Graph Construction
"""
src/data_pipeline.py
Downloads the human PPI network AND sequences from STRING DB, 
samples 100,000 edges, generates REAL ProtBERT embeddings, 
and builds a PyTorch Geometric Data object.
"""
import h5py
import numpy as np
import re
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
    print("Loading edges...")
    edges_df = pd.read_csv(
        os.path.join(raw_dir, "9606.protein.links.detailed.v12.0.txt"), 
        sep=' ', nrows=num_edges
    )
    
    # We only care about high confidence experimental/database links for Alzheimer's
    edges_df = edges_df[(edges_df['experimental'] > 400) | (edges_df['database'] > 400)]
    
    unique_proteins = set(edges_df['protein1']).union(set(edges_df['protein2']))
    protein_to_idx = {prot: i for i, prot in enumerate(unique_proteins)}
    
    print(f"Graph contains {len(unique_proteins)} unique proteins.")
    
    # Initialize a tensor for embeddings (ProtT5 uses 1024 dimensions)
    x = torch.zeros((len(unique_proteins), 1024))
    missing_proteins = 0
    
    print("Extracting pre-computed ProtT5 embeddings...")
    
    # Load the HDF5 file containing the pre-computed embeddings
    h5_path = os.path.join(raw_dir, "9606.protein.sequence.embeddings.v12.0.h5")
    
    with h5py.File(h5_path, 'r') as h5_file:
        for protein_id, idx in protein_to_idx.items():
            # STRING h5 files typically store arrays with the protein ID as the dataset key
            if protein_id in h5_file:
                # Extract the embedding, convert to tensor, and assign to the node row
                embedding_array = np.array(h5_file[protein_id])
                x[idx] = torch.tensor(embedding_array)
            else:
                missing_proteins += 1
                # If a protein is missing, it remains a zero-vector
                
    print(f"Embeddings loaded. {missing_proteins} proteins lacked pre-computed embeddings.")

    # 3. Build edge_index tensor
    print("Building edge connections...")
    src = [protein_to_idx[p] for p in edges_df['protein1']]
    dst = [protein_to_idx[p] for p in edges_df['protein2']]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 4. Save the graph object
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, os.path.join(processed_dir, "alzheimers_ppi_graph.pt"))
    print("Graph built and saved successfully!")
    
    print(f"Generating embeddings for {num_nodes} unique proteins...")
    # We sort by index so the tensor perfectly matches the edge_index mapping
    for protein, idx in tqdm(sorted(protein_to_idx.items(), key=lambda item: item[1])):
        raw_sequence = protein_sequences.get(protein, "")
        
        if raw_sequence:
            # Ensure the raw sequence is a string and force the extraction
            # We skip the ID/Sequence logic entirely since we know it's a sequence from FASTA
            spaced_sequence = " ".join(list(re.sub(r"[UZOB]", "X", raw_sequence.upper())))
            
            encoded = extractor.tokenizer(
                spaced_sequence, return_tensors='pt', padding=True, truncation=True, max_length=1024
            ).to(extractor.device)
            
            with torch.no_grad():
                out = extractor.model(**encoded)
                embedding = torch.mean(out.last_hidden_state, dim=1)
                
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
