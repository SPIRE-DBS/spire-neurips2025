#basic reconstruction and loss evaluation
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def evaluate_spire(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon_xb, recon_yb, shared_xb, shared_yb, _ ,_ = model(xb, yb)
            break  # visualize just one batch

    mse = F.mse_loss(recon_yb, yb).item()
    r2 = 1 - torch.sum((yb - recon_yb)**2) / torch.sum((yb - torch.mean(yb))**2)

    plt.figure(figsize=(14, 4))
    plt.plot(yb.cpu().flatten(), label='True STN')
    plt.plot(recon_yb.cpu().flatten(), label='Reconstructed STN', linestyle='--')
    plt.title("SPIRE Model: True vs Reconstructed STN")
    plt.xlabel("Time (flattened)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"📊 Validation MSE: {mse:.4f}")
    print(f"📈 Validation R² Score: {r2:.4f}")