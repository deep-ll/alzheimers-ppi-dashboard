# Main Streamlit Application Entrypoint
# app.py
import streamlit as st
import torch
import wandb
import os
from src.model import GraphSAGEEncoder, EdgeDecoder
from src.feature_extractor import ProteinFeatureExtractor

# 1. Page Setup
st.set_page_config(page_title="Alzheimer's PPI Predictor", page_icon="🧬")
st.title("🧬 Alzheimer's Protein Interaction Predictor")
st.write("Powered by GraphSAGE & ProtBERT")

# 2. Load Models Safely (Cached so it doesn't reload on every button click)
@st.cache_resource
def load_pipeline():
    # Streamlit uses st.secrets to securely hold your passwords
    wandb.login(key=st.secrets["WANDB_API_KEY"])
    run = wandb.init(project="alzheimers-ppi-graphsage", job_type="inference")
    
    # Download the best model from your W&B Registry
    artifact = run.use_artifact('your-username/alzheimers-ppi-graphsage/graphsage-ppi-model:latest')
    artifact_dir = artifact.download()
    
    # Initialize your architecture
    model = GraphSAGEEncoder(in_channels=1024, hidden_channels=64, out_channels=32)
    edge_decoder = EdgeDecoder()
    
    # Load the weights
    checkpoint = torch.load(f"{artifact_dir}/best_model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    edge_decoder.load_state_dict(checkpoint['decoder_state'])
    
    model.eval()
    edge_decoder.eval()
    
    # Initialize the feature extractor
    extractor = ProteinFeatureExtractor()
    
    return model, edge_decoder, extractor

# Start loading
with st.spinner("Loading AI Models from Weights & Biases..."):
    model, edge_decoder, extractor = load_pipeline()

# 3. The User Interface
protein_A = st.text_input("Enter Protein A (e.g., APP, P05067, or sequence)")
protein_B = st.text_input("Enter Protein B (e.g., MAPT, P10636, or sequence)")

if st.button("Predict Interaction"):
    if protein_A and protein_B:
        with st.spinner("Extracting embeddings and running GraphSAGE..."):
            try:
                # Get embeddings
                feat_A = extractor.get_embedding(protein_A)
                feat_B = extractor.get_embedding(protein_B)
                
                # Format for PyG
                x = torch.cat([feat_A, feat_B], dim=0)
                edge_index = torch.tensor([[0], [1]], dtype=torch.long)
                edge_label_index = torch.tensor([[0], [1]], dtype=torch.long)
                
                # Predict
                with torch.no_grad():
                    z = model(x, edge_index)
                    logits = edge_decoder(z, edge_label_index)
                    probability = torch.sigmoid(logits).item()
                
                st.success(f"### Interaction Probability: {probability:.2%}")
                
            except Exception as e:
                st.error(f"Error processing proteins: {str(e)}")
    else:
        st.warning("Please enter both proteins.")
