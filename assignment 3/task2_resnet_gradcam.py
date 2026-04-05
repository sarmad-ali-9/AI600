"""
Task 2: Transfer Learning (ResNet-18 on STL-10) + GradCAM Visualisation
AI600 Deep Learning – Assignment 3, LUMS Spring 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import cv2

matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('outputs', exist_ok=True)

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def get_stl10_loaders(batch_size=64):
    # STL-10 images are 96×96 RGB
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = torchvision.datasets.STL10(
        root='./data', split='train', download=True, transform=train_tf)
    test_ds  = torchvision.datasets.STL10(
        root='./data', split='test',  download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, train_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes=10):
    """
    Pre-trained ResNet-18 with:
    - All convolutional layers (backbone) frozen.
    - Final FC layer replaced with Linear(512, num_classes).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Replace head – only this will be trained
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Count params
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}  (head only)")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label='Train')
    axes[0].plot(epochs, val_losses,   label='Test')
    axes[0].set_title(f'{title} – Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label='Train')
    axes[1].plot(epochs, val_accs,   label='Test')
    axes[1].set_title(f'{title} – Accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved plot: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Uses a forward hook to capture activations and registers a gradient hook
    on the activation tensor itself (tensor-level hook) so that gradients are
    captured reliably regardless of the module type.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._fwd_hook   = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        # Force the output tensor to require gradients so the hook fires
        # even when backbone parameters are frozen.
        self.activations = output.requires_grad_(True)
        self.activations.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.to(device)

        # Forward
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward for target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients over spatial dims → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        self._fwd_hook.remove()


def denormalise(tensor):
    """Convert normalised ImageNet tensor to [0,1] numpy image."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    return np.clip(img, 0, 1)


def overlay_cam(img_np, cam_np, alpha=0.4):
    """Overlay GradCAM heatmap on RGB image."""
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam_np, (w, h))

    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = alpha * heatmap + (1 - alpha) * img_np
    return np.clip(overlay, 0, 1)


def visualise_gradcam(model, test_ds, num_correct=2, num_wrong=2):
    """
    Select 2 correctly and 2 incorrectly classified images from the test set.
    Generate and save GradCAM overlays.
    """
    gradcam = GradCAM(model, model.layer4[-1])

    correct_samples, wrong_samples = [], []
    model.eval()

    for idx in range(len(test_ds)):
        if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
            break

        img, label = test_ds[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()

        if pred == label and len(correct_samples) < num_correct:
            correct_samples.append((img, label, pred))
        elif pred != label and len(wrong_samples) < num_wrong:
            wrong_samples.append((img, label, pred))

    samples = correct_samples + wrong_samples
    labels  = (['Correct'] * len(correct_samples) +
                ['Wrong']   * len(wrong_samples))

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = axes[:, np.newaxis]

    for col, ((img, label, pred), verdict) in enumerate(zip(samples, labels)):
        cam, _ = gradcam(img.unsqueeze(0), class_idx=pred)
        img_np  = denormalise(img)
        overlay = overlay_cam(img_np, cam)

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(
            f"GT: {STL10_CLASSES[label]}\nPred: {STL10_CLASSES[pred]}\n({verdict})",
            fontsize=9)
        axes[0, col].axis('off')

        axes[1, col].imshow(overlay)
        axes[1, col].set_title('GradCAM', fontsize=9)
        axes[1, col].axis('off')

    plt.suptitle('GradCAM Visualisations – ResNet-18 on STL-10', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/gradcam.png', dpi=150)
    plt.close()
    print("Saved GradCAM figure: outputs/gradcam.png")

    gradcam.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_task2(num_epochs=20, batch_size=64, lr=1e-3):
    print("\n" + "="*60)
    print("TASK 2A: ResNet-18 Fine-tuning on STL-10")
    print("="*60)

    train_loader, test_loader, _, test_ds = get_stl10_loaders(batch_size)
    model = build_model(num_classes=10)

    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    for epoch in range(1, num_epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(tl); test_losses.append(vl)
        train_accs.append(ta);   test_accs.append(va)
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {tl:.4f} Acc: {ta:.4f} | "
              f"Test  Loss: {vl:.4f} Acc: {va:.4f}")

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {final_test_acc*100:.2f}%")

    plot_curves(train_losses, test_losses, train_accs, test_accs,
                'ResNet-18 STL-10', 'outputs/stl10_curves.png')

    torch.save(model.state_dict(), 'outputs/stl10_model.pth')

    # ── GradCAM ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TASK 2B: GradCAM Visualisation")
    print("="*60)

    _, _, _, test_ds_raw = get_stl10_loaders(batch_size)
    visualise_gradcam(model, test_ds_raw)

    return model, train_losses, test_losses, train_accs, test_accs, final_test_acc


if __name__ == '__main__':
    run_task2()
