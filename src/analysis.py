"""Visualization and analysis tools: confusion matrix, Grad-CAM, t-SNE."""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from sklearn.manifold import TSNE

from src.config import DEVICE, NUM_CLASSES, CLASS_LIST, OUTPUT_DIR


def load_class_names():
    """Load class names from class_list_food.txt."""
    names = {}
    with open(CLASS_LIST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                names[int(parts[0]) - 1] = parts[1]  # 0-indexed
    return names


CLASS_NAMES = load_class_names()


# ── Training curves ────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, history["lr"])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


# ── Confusion matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(cm, save_path, top_n=20):
    """Plot confusion matrix (top N most confused classes)."""
    # Find the most confused pairs
    np.fill_diagonal(cm, 0)
    confused = np.unravel_index(np.argsort(cm.ravel())[::-1][:top_n], cm.shape)
    involved_classes = sorted(set(confused[0].tolist() + confused[1].tolist()))[:top_n]

    sub_cm = cm[np.ix_(involved_classes, involved_classes)]
    names = [CLASS_NAMES.get(c, str(c))[:15] for c in involved_classes]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub_cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (top {top_n} confused classes)")
    plt.colorbar(im)

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(names)):
            if sub_cm[i, j] > 0:
                ax.text(j, i, str(sub_cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ── Per-class accuracy ────────────────────────────────────────────────

def plot_per_class_accuracy(per_class_acc, save_path, train_counts=None):
    """Bar chart of per-class accuracy, optionally colored by train count."""
    classes = sorted(per_class_acc.keys())
    accs = [per_class_acc[c] for c in classes]
    names = [CLASS_NAMES.get(c, str(c)) for c in classes]

    # Sort by accuracy
    sorted_idx = np.argsort(accs)
    names = [names[i] for i in sorted_idx]
    accs = [accs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 20))

    if train_counts is not None:
        counts = [train_counts.get(classes[i], 0) for i in sorted_idx]
        colors = plt.cm.RdYlGn(np.array(counts) / max(counts))
        bars = ax.barh(range(len(names)), accs, color=colors)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(min(counts), max(counts)))
        plt.colorbar(sm, ax=ax, label="Train samples")
    else:
        bars = ax.barh(range(len(names)), accs, color="steelblue")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Validation Accuracy")
    ax.axvline(np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy saved to {save_path}")


# ── Grad-CAM ──────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM visualization for CNN models."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for a single image."""
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE) if input_tensor.dim() == 3 else input_tensor.to(DEVICE)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


def plot_gradcam(images, model, target_layer, save_path, dataset_mean, dataset_std, n=8):
    """Generate Grad-CAM visualizations for a batch of images."""
    gradcam = GradCAM(model, target_layer)
    mean = torch.tensor(dataset_mean).view(3, 1, 1)
    std = torch.tensor(dataset_std).view(3, 1, 1)

    n = min(n, len(images))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i in range(n):
        img = images[i]
        cam, pred_class = gradcam.generate(img)

        # Denormalize for display
        img_display = img.cpu() * std + mean
        img_display = img_display.permute(1, 2, 0).numpy().clip(0, 1)

        axes[0, i].imshow(img_display)
        axes[0, i].set_title(CLASS_NAMES.get(pred_class, str(pred_class))[:20], fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(img_display)
        axes[1, i].imshow(cam, alpha=0.5, cmap="jet")
        axes[1, i].set_title("Grad-CAM", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle("Grad-CAM Visualizations", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM saved to {save_path}")


# ── t-SNE feature visualization ──────────────────────────────────────

@torch.no_grad()
def compute_features(model, dataloader, max_samples=2000):
    """Extract features from model for t-SNE."""
    model.eval()
    features, labels = [], []
    count = 0

    for imgs, lbls in dataloader:
        if count >= max_samples:
            break
        imgs = imgs.to(DEVICE)
        if hasattr(model, "get_features"):
            feats = model.get_features(imgs)
        else:
            # Fallback: use output logits
            feats = model(imgs)
        features.append(feats.cpu().numpy())
        labels.append(lbls.numpy())
        count += imgs.size(0)

    features = np.concatenate(features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]
    return features, labels


def plot_tsne(features, labels, save_path, perplexity=30):
    """Create t-SNE visualization of learned features."""
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(14, 12))
    scatter = ax.scatter(
        embedded[:, 0], embedded[:, 1],
        c=labels, cmap="tab20", alpha=0.6, s=8,
    )
    ax.set_title("t-SNE of Learned Features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE saved to {save_path}")


# ── Experiment comparison ─────────────────────────────────────────────

def plot_experiment_comparison(results_dict, save_path):
    """Bar chart comparing final val accuracy across experiments."""
    names = list(results_dict.keys())
    accs = [results_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), accs, color="steelblue")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Experiment Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved to {save_path}")
