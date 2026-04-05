"""
Task 1A & 1B: Custom CNN for Standard MNIST and Colored-MNIST (C-MNIST)
AI600 Deep Learning - Assignment 3, LUMS Spring 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import io
import zipfile

matplotlib.use('Agg')

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    """
    Custom CNN for MNIST / C-MNIST.
    Architecture (1-channel grayscale version):
      Conv1:  in_ch → 8,  3×3, ReLU, MaxPool 2×2
      Conv2:  8     → 16, 3×3, ReLU, MaxPool 2×2
      Conv3:  16    → 32, 3×3, ReLU
      FC1:    288   → 64, ReLU
      FC2:    64    → 10

    Parameter count (1-channel): ~25 k  (well under 50 k limit)
    Parameter count (3-channel): ~25.2 k
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=0),  # 28→26
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 26→13

            # Conv 2
            nn.Conv2d(8, 16, kernel_size=3, padding=0),            # 13→11
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 11→5

            # Conv 3
            nn.Conv2d(16, 32, kernel_size=3, padding=0),           # 5→3
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label='Train')
    axes[0].plot(epochs, val_losses,   label='Val')
    axes[0].set_title(f'{title} – Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label='Train')
    axes[1].plot(epochs, val_accs,   label='Val')
    axes[1].set_title(f'{title} – Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved plot: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1A: Standard MNIST
# ─────────────────────────────────────────────────────────────────────────────
def run_mnist(num_epochs=15, batch_size=128, lr=1e-3, val_split=0.1):
    print("\n" + "="*60)
    print("TASK 1A: Standard MNIST")
    print("="*60)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=tf)
    test_ds    = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=tf)

    n_val  = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    model = TinyCNN(in_channels=1).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, num_epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = evaluate(model, val_loader, criterion)
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {tl:.4f} Acc: {ta:.4f} | "
              f"Val   Loss: {vl:.4f} Acc: {va:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

    plot_curves(train_losses, val_losses, train_accs, val_accs,
                'MNIST', 'outputs/mnist_curves.png')

    # ── Visualise first-layer filters ──────────────────────────────────────
    filters = model.conv_block[0].weight.data.cpu()  # shape: [8, 1, 3, 3]
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i, ax in enumerate(axes):
        f = filters[i, 0]
        f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f_norm, cmap='gray', interpolation='nearest')
        ax.set_title(f'F{i+1}', fontsize=8)
        ax.axis('off')
    plt.suptitle('Conv1 Filters – Standard MNIST', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/mnist_filters.png', dpi=150)
    plt.close()
    print("Saved plot: outputs/mnist_filters.png")

    torch.save(model.state_dict(), 'outputs/mnist_model.pth')
    return model, train_losses, val_losses, train_accs, val_accs, test_acc


# ─────────────────────────────────────────────────────────────────────────────
# Task 1B: Colored-MNIST (C-MNIST)
# ─────────────────────────────────────────────────────────────────────────────
def load_cmnist_zip(zip_path='cmnist.zip'):
    """
    Load C-MNIST from the LMS-provided zip file.
    The zip contains: train_biased.pt, test_biased.pt, test_unbiased.pt
    Each .pt file is a tuple (images, labels) where images are float32 [N,3,28,28]
    in the range [0, 1].
    """
    with zipfile.ZipFile(zip_path) as z:
        def load_pt(name):
            imgs, labels = torch.load(io.BytesIO(z.read(name)), weights_only=False)
            # Normalize from [0,1] to [-1,1] to match standard practice
            imgs = imgs * 2.0 - 1.0
            return TensorDataset(imgs, labels)

        train_ds        = load_pt('train_biased.pt')
        test_biased_ds  = load_pt('test_biased.pt')
        test_unbiased_ds= load_pt('test_unbiased.pt')
    return train_ds, test_biased_ds, test_unbiased_ds


def run_cmnist(num_epochs=15, batch_size=128, lr=1e-3, val_split=0.1,
               cmnist_zip='cmnist.zip'):
    print("\n" + "="*60)
    print("TASK 1B: Colored MNIST (C-MNIST)")
    print("="*60)

    if not os.path.isfile(cmnist_zip):
        print(f"[SKIP] C-MNIST zip not found at '{cmnist_zip}'.")
        return None

    full_train, test_biased_ds, test_unbiased_ds = load_cmnist_zip(cmnist_zip)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader    = DataLoader(train_ds,          batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader      = DataLoader(val_ds,            batch_size=batch_size, shuffle=False, num_workers=2)
    biased_loader   = DataLoader(test_biased_ds,    batch_size=batch_size, shuffle=False, num_workers=2)
    unbiased_loader = DataLoader(test_unbiased_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    model = TinyCNN(in_channels=3).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, num_epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = evaluate(model, val_loader, criterion)
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {tl:.4f} Acc: {ta:.4f} | "
              f"Val   Loss: {vl:.4f} Acc: {va:.4f}")

    _, biased_acc   = evaluate(model, biased_loader,   criterion)
    _, unbiased_acc = evaluate(model, unbiased_loader, criterion)
    print(f"\nBiased   Test Accuracy: {biased_acc*100:.2f}%")
    print(f"Unbiased Test Accuracy: {unbiased_acc*100:.2f}%")

    plot_curves(train_losses, val_losses, train_accs, val_accs,
                'C-MNIST', 'outputs/cmnist_curves.png')

    torch.save(model.state_dict(), 'outputs/cmnist_model.pth')
    return model, train_losses, val_losses, train_accs, val_accs, biased_acc, unbiased_acc


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_mnist()
    run_cmnist()
