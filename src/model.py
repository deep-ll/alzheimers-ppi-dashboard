# GraphSAGE & EdgeDecoder Models
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # First GAT layer (multi-head attention)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # Second GAT layer (single head for final embedding)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def encode(self, x, edge_index):
        """Generates node embeddings"""
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """Predicts edge existence using dot product of node embeddings"""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # Returns raw logits; we'll use BCEWithLogitsLoss during training
        return (src * dst).sum(dim=-1)
