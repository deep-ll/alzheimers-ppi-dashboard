import os
import torch
import pandas as pd
from src.model import GATLinkPredictor

class PPIInference:
    def __init__(self, model_path, graph_path, mapping_path):
        # We use CPU for Streamlit since it's cheap and inference for one edge is instant
        self.device = torch.device('cpu')
        
        # 1. Load protein ID to Node Index mapping
        self.mapping_df = pd.read_csv(mapping_path)
        self.id_to_idx = dict(zip(self.mapping_df['string_protein_id'], self.mapping_df['node_idx']))
        self.available_proteins = sorted(list(self.id_to_idx.keys()))
        
        # 2. Load Graph
        self.graph_data = torch.load(graph_path, weights_only=False).to(self.device)
        
        # 3. Load Model
        self.model = GATLinkPredictor(in_channels=1024, hidden_channels=128, out_channels=64, heads=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Pre-compute all node embeddings so web requests are lightning fast
        with torch.no_grad():
            self.node_embeddings = self.model.encode(self.graph_data.x, self.graph_data.edge_index)
            
    def predict_interaction(self, protein1, protein2):
        if protein1 not in self.id_to_idx or protein2 not in self.id_to_idx:
            return None
            
        idx1 = self.id_to_idx[protein1]
        idx2 = self.id_to_idx[protein2]
        
        # Format the query as a PyTorch Geometric edge_index
        edge_query = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logit = self.model.decode(self.node_embeddings, edge_query)
            probability = torch.sigmoid(logit).item()
            
        return probability
