"""Food Recognition dataset and data loading utilities."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

from src.config import (
    TRAIN_IMG_DIR, TEST_IMG_DIR, TRAIN_LABELS_CSV,
    DATASET_MEAN, DATASET_STD, NUM_CLASSES,
    BATCH_SIZE, NUM_WORKERS, VAL_SPLIT, SEED, IMG_SIZE,
)


# ── Augmentation presets ───────────────────────────────────────────────

def get_transforms(mode="train", img_size=None, augmentation="basic"):
    """Return transforms for train/val/test.

    augmentation levels:
        'none'    – only resize + normalize (for val/test)
        'basic'   – flip + small crop
        'medium'  – + color jitter + rotation
        'heavy'   – + random erasing + affine
    """
    sz = img_size or IMG_SIZE
    normalize = transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)

    if mode in ("val", "test"):
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            normalize,
        ])

    # Train transforms
    aug_list = [transforms.Resize((sz + 16, sz + 16))]

    if augmentation in ("basic", "medium", "heavy"):
        aug_list += [
            transforms.RandomCrop(sz),
            transforms.RandomHorizontalFlip(),
        ]
    else:  # 'none'
        aug_list += [transforms.CenterCrop(sz)]

    if augmentation in ("medium", "heavy"):
        aug_list += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
        ]

    if augmentation == "heavy":
        aug_list += [
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomGrayscale(p=0.05),
        ]

    aug_list += [transforms.ToTensor(), normalize]

    if augmentation == "heavy":
        aug_list.append(transforms.RandomErasing(p=0.2))

    return transforms.Compose(aug_list)


# ── Dataset class ──────────────────────────────────────────────────────

class FoodDataset(Dataset):
    """Dataset for food recognition images."""

    def __init__(self, img_paths, labels=None, transform=None, cache_in_memory=False):
        self.img_paths = img_paths
        self.labels = labels  # None for test set
        self.transform = transform
        self.cache = {}
        self.cache_in_memory = cache_in_memory

        if cache_in_memory:
            print(f"  Caching {len(img_paths)} images in memory...")
            for i, p in enumerate(img_paths):
                self.cache[i] = Image.open(p).convert("RGB")
                if (i + 1) % 5000 == 0:
                    print(f"    {i+1}/{len(img_paths)} cached")
            print(f"    Done caching.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.cache:
            img = self.cache[idx]
        else:
            img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            # Labels in CSV are 1-80, convert to 0-79 for PyTorch
            label = self.labels[idx] - 1
            return img, label
        return img


# ── Data loading helpers ───────────────────────────────────────────────

def load_train_data():
    """Load training image paths and labels from CSV."""
    img_paths, labels = [], []
    with open(TRAIN_LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_paths.append(TRAIN_IMG_DIR / row["img_name"])
            labels.append(int(row["label"]))
    return img_paths, labels


def load_test_data():
    """Load test image paths (sorted by index)."""
    import re
    test_files = sorted(
        TEST_IMG_DIR.iterdir(),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    img_names = [p.name for p in test_files]
    return test_files, img_names


def get_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced classes."""
    counts = np.bincount(np.array(labels) - 1, minlength=NUM_CLASSES).astype(np.float32)
    # Avoid division by zero; use inverse frequency
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES  # normalize so mean=1
    return torch.FloatTensor(weights)


def get_dataloaders(
    img_size=None,
    batch_size=None,
    augmentation="basic",
    use_weighted_sampler=True,
):
    """Create train/val dataloaders with stratified split."""
    bs = batch_size or BATCH_SIZE
    img_paths, labels = load_train_data()

    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=VAL_SPLIT, random_state=SEED, stratify=labels,
    )

    train_tf = get_transforms("train", img_size, augmentation)
    val_tf = get_transforms("val", img_size)

    train_ds = FoodDataset(train_paths, train_labels, train_tf, cache_in_memory=True)
    val_ds = FoodDataset(val_paths, val_labels, val_tf, cache_in_memory=True)

    # Weighted sampler to handle class imbalance
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_weights = get_class_weights(train_labels)
        sample_weights = class_weights[torch.LongTensor(train_labels) - 1]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False  # sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


def get_test_loader(img_size=None, batch_size=None):
    """Create test dataloader."""
    bs = batch_size or BATCH_SIZE
    test_paths, test_names = load_test_data()
    test_tf = get_transforms("test", img_size)
    test_ds = FoodDataset(test_paths, transform=test_tf)
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return test_loader, test_names
