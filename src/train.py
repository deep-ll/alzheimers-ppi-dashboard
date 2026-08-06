# Training Loop & W&B Artifact Logging
import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

# Ensure correct path for imports in Colab
import sys
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from src.model import GATLinkPredictor

def train(model, optimizer, criterion, train_data):
    model.train()
    optimizer.zero_grad()
    
    # 1. Generate node embeddings
    z = model.encode(train_data.x, train_data.edge_index)
    
    # 2. Predict links
    predictions = model.decode(z, train_data.edge_label_index)
    
    # 3. Calculate loss
    loss = criterion(predictions, train_data.edge_label.float())
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    predictions = model.decode(z, data.edge_label_index)
    
    # Convert logits to probabilities
    probs = torch.sigmoid(predictions).cpu().numpy()
    labels = data.edge_label.cpu().numpy()
    
    # Calculate ROC-AUC
    return roc_auc_score(labels, probs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Graph
    processed_dir = "/content/drive/MyDrive/Alzheimers_PPI_Project/processed"
    graph_path = os.path.join(processed_dir, "pyg_ppi_graph.pt")
    
    print("Loading graph data...")
    data = torch.load(graph_path)
    
    # Split edges into Train (70%), Val (10%), Test (20%)
    # This automatically adds negative samples for link prediction
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=True,
        num_val=0.1,
        num_test=0.2
    )
    train_data, val_data, test_data = transform(data)
    
    # Move splits to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Initialize Model (Input: 1024 from ProtT5, Hidden: 128, Output: 64)
    model = GATLinkPredictor(in_channels=1024, hidden_channels=128, out_channels=64, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...\n")
    epochs = 100
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, criterion, train_data)
        
        if epoch % 10 == 0:
            val_auc = test(model, val_data)
            print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    # Final Test
    test_auc = test(model, test_data)
    print(f"\n✅ Training Complete! Final Test ROC-AUC: {test_auc:.4f}")
    
    # Save Model
    model_path = os.path.join(processed_dir, "gat_link_predictor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
