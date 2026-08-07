import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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

def plot_metrics(train_losses, val_aucs, save_path):
    """Generates and saves a plot of the training curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Validation AUC
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Validation AUC', color='orange', linewidth=2)
    plt.title('Validation ROC-AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Graph
    processed_dir = "/content/drive/MyDrive/Alzheimers_PPI_Project/processed"
    graph_path = os.path.join(processed_dir, "pyg_ppi_graph.pt")
    
    print("Loading graph data...")
    # FIXED: Added weights_only=False for PyTorch 2.6+
    data = torch.load(graph_path, weights_only=False) 
    
    print(f"Graph loaded with {data.num_nodes} nodes and {data.num_edges} edges.")
    
    # Split edges into Train (70%), Val (10%), Test (20%)
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=True,
        num_val=0.1,
        num_test=0.2
    )
    train_data, val_data, test_data = transform(data)
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Initialize Model
    model = GATLinkPredictor(in_channels=1024, hidden_channels=128, out_channels=64, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...\n")
    epochs = 100
    train_losses = []
    val_aucs = []

    for epoch in range(1, epochs + 1):
        # Train and Evaluate
        loss = train(model, optimizer, criterion, train_data)
        val_auc = test(model, val_data)
        
        # Store metrics for plotting
        train_losses.append(loss)
        val_aucs.append(val_auc)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    # Final Test
    test_auc = test(model, test_data)
    print(f"\n✅ Training Complete! Final Test ROC-AUC: {test_auc:.4f}")
    
    # Save Model
    model_path = os.path.join(processed_dir, "gat_link_predictor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save Plot
    plot_path = os.path.join(processed_dir, "training_metrics.png")
    plot_metrics(train_losses, val_aucs, plot_path)
    print(f"Metrics curve saved to {plot_path}")

if __name__ == "__main__":
    main()
