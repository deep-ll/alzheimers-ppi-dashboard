import streamlit as st
import os
from src.inference import PPIInference

# Configure the web page
st.set_page_config(page_title="Alzheimer's PPI Predictor", layout="centered")

st.title("🧬 Alzheimer's Protein Interaction Predictor")
st.markdown("Use Graph Neural Networks (GAT) to predict undiscovered protein interactions.")

# Cache the model so it only loads once
@st.cache_resource
def load_predictor():
    # Use relative paths for GitHub hosting!
    processed_dir = "data" 
    
    return PPIInference(
        model_path=os.path.join(processed_dir, "gat_link_predictor.pth"),
        graph_path=os.path.join(processed_dir, "pyg_ppi_graph.pt"),
        mapping_path=os.path.join(processed_dir, "node_mapping.csv")
    )

try:
    with st.spinner("Loading AI Model and 14,000+ proteins... (This takes a few seconds on boot)"):
        predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout: Two columns for selecting proteins
col1, col2 = st.columns(2)

with col1:
    p1 = st.selectbox("Select Protein 1 (STRING ID):", predictor.available_proteins, index=0)
with col2:
    p2 = st.selectbox("Select Protein 2 (STRING ID):", predictor.available_proteins, index=1)

# Prediction Button
if st.button("Predict Interaction", use_container_width=True):
    if p1 == p2:
        st.warning("Please select two different proteins.")
    else:
        probability = predictor.predict_interaction(p1, p2)
        
        if probability is not None:
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Display colored metrics based on confidence
            pct = probability * 100
            if probability > 0.80:
                st.success(f"**High Confidence:** There is a {pct:.2f}% chance these proteins interact.")
            elif probability > 0.50:
                st.info(f"**Moderate Confidence:** There is a {pct:.2f}% chance these proteins interact.")
            else:
                st.warning(f"**Low Confidence:** There is a {pct:.2f}% chance these proteins interact.")
        else:
            st.error("Error finding proteins in the database.")
