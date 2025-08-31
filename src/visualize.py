import torch
import matplotlib.pyplot as plt

def plot_predictions(model, dataloader, device, num_images=20):
    """Plots model predictions for a few validation images."""
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(4, 5, i + 1)
        img = images[i].cpu().squeeze() * 0.5 + 0.5
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}",
                  color=("green" if preds[i] == labels[i] else "red"))
        plt.axis('off')
    plt.tight_layout()
    plt.show()