"""Training and evaluation framework with logging."""

import time
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR

from src.config import DEVICE, EPOCHS, LR, WEIGHT_DECAY, OUTPUT_DIR, NUM_CLASSES
from src.dataset import get_class_weights


class Trainer:
    """Handles training, validation, logging, and checkpointing."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        experiment_name="experiment",
        epochs=None,
        lr=None,
        weight_decay=None,
        scheduler_type="cosine",  # 'cosine', 'step', 'onecycle', 'none'
        use_class_weights=True,
        label_smoothing=0.0,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_name = experiment_name
        self.epochs = epochs or EPOCHS
        self.lr = lr or LR

        # Loss function with optional class weights and label smoothing
        if use_class_weights:
            # Use raw labels directly (avoid iterating dataset which loads all images)
            all_labels = train_loader.dataset.labels
            weights = get_class_weights(all_labels).to(DEVICE)
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        wd = weight_decay if weight_decay is not None else WEIGHT_DECAY
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=wd)

        # LR scheduler
        self.scheduler = self._get_scheduler(scheduler_type)

        # Logging
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        self.best_val_acc = 0.0
        self.exp_dir = OUTPUT_DIR / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _get_scheduler(self, scheduler_type):
        if scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        elif scheduler_type == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer, max_lr=self.lr,
                steps_per_epoch=len(self.train_loader), epochs=self.epochs,
            )
        return None

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

    def train(self):
        """Full training loop with logging and checkpointing."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {DEVICE}")
        print(f"Epochs: {self.epochs}, LR: {self.lr}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()

            # Step scheduler (except OneCycle which steps per batch)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | {elapsed:.1f}s"
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.exp_dir / "best_model.pt")
                print(f"  → New best val accuracy: {val_acc:.4f}")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} min. Best val acc: {self.best_val_acc:.4f}")

        # Save training history
        with open(self.exp_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Save final model
        torch.save(self.model.state_dict(), self.exp_dir / "final_model.pt")

        return self.history

    @torch.no_grad()
    def predict_test(self, test_loader, test_names):
        """Generate predictions for the test set."""
        self.model.eval()
        all_preds = []

        for imgs in test_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(DEVICE)
            outputs = self.model(imgs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())

        # Convert back to 1-80 labels for submission
        all_preds = [p + 1 for p in all_preds]
        return all_preds

    def save_submission(self, test_loader, test_names, filename=None):
        """Generate and save submission CSV."""
        preds = self.predict_test(test_loader, test_names)
        fname = filename or f"submission_{self.experiment_name}.csv"
        path = self.exp_dir / fname

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["img_name", "label"])
            for name, pred in zip(test_names, preds):
                writer.writerow([name, pred])

        print(f"Submission saved to {path}")
        return path

    @torch.no_grad()
    def get_confusion_matrix(self):
        """Compute confusion matrix on validation set."""
        _, _, preds, labels = self.validate()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for p, l in zip(preds, labels):
            cm[l, p] += 1
        return cm

    @torch.no_grad()
    def get_per_class_accuracy(self):
        """Compute per-class accuracy on validation set."""
        _, _, preds, labels = self.validate()
        per_class = {}
        for c in range(NUM_CLASSES):
            mask = labels == c
            if mask.sum() > 0:
                per_class[c] = (preds[mask] == c).mean()
            else:
                per_class[c] = 0.0
        return per_class
