import torch
import torch.nn as nn
from config import config
from src.model import VisionTransformer
from src.dataset import get_data_loaders
from src.training import train_one_epoch, evaluate
from src.visualize import plot_predictions

def main():
    # Load data
    train_loader, val_loader = get_data_loaders(config["batch_size"])
    
    # Instantiate the model
    model = VisionTransformer(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_channels=config["num_channels"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        num_layers=config["num_transformer_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(config["device"])
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated on {config['device'].upper()}.")
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    print("\nStarting training...")
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config["device"])
        val_loss, val_acc = evaluate(model, val_loader, criterion, config["device"])
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print("\nTraining finished.")
    

    print("\nPlotting predictions from validation set...")
    plot_predictions(model, val_loader, config["device"])

if __name__ == '__main__':
    main()