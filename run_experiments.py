"""
Food Recognition Challenge 2026 — Full Experiment Pipeline
============================================================
Research question:
    Which combination of CNN architecture, data augmentation, and
    regularization yields the highest fine-grained food classification
    accuracy without pretrained weights?

Experiments:
    1. Baseline CNN (simple 4-layer conv)
    2. Deeper CNN + BatchNorm
    3. ResNet-style skip connections
    4. ResNet + CBAM Attention
    5. Augmentation ablation (none / basic / medium / heavy)
    6. LR schedule comparison (step / cosine / onecycle)
    7. Label smoothing effect

Usage:
    python run_experiments.py                    # run all experiments
    python run_experiments.py --exp baseline     # run single experiment
    python run_experiments.py --exp augmentation # run augmentation ablation
    python run_experiments.py --analyze          # only run analysis on saved models
"""

import argparse
import json
import os
import sys
import gc
import torch
import numpy as np
from collections import Counter

# MPS stability fixes: fallback to CPU for unsupported ops, disable memory cap
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def flush():
    """Flush stdout so output is visible in real-time."""
    sys.stdout.flush()


def mps_cleanup():
    """Free MPS GPU memory between experiments to prevent hangs."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()

from src.config import OUTPUT_DIR, DEVICE, NUM_CLASSES, DATASET_MEAN, DATASET_STD
from src.dataset import get_dataloaders, get_test_loader, load_train_data
from src.models import get_model
from src.trainer import Trainer
from src.analysis import (
    plot_training_curves, plot_confusion_matrix, plot_per_class_accuracy,
    plot_gradcam, compute_features, plot_tsne, plot_experiment_comparison,
    load_class_names,
)


def run_architecture_experiments():
    """Experiment 1-4: Compare architectures."""
    results = {}

    configs = [
        ("1_baseline", "baseline", {"augmentation": "basic", "img_size": 128}, {"epochs": 20, "lr": 1e-3}),
        ("2_deep_cnn", "deep_cnn", {"augmentation": "basic", "img_size": 128}, {"epochs": 20, "lr": 1e-3}),
        ("3_resnet", "resnet", {"augmentation": "basic", "img_size": 128}, {"epochs": 20, "lr": 1e-3}),
        ("4_attention_resnet", "attention_resnet", {"augmentation": "basic", "img_size": 128}, {"epochs": 20, "lr": 1e-3}),
    ]

    # Pre-load train counts once (avoid repeated CSV reads)
    _, all_labels = load_train_data()
    train_counts = dict(Counter([l - 1 for l in all_labels]))

    # Cache dataloaders: all arch experiments use same data config
    shared_loaders = None

    for exp_name, model_name, data_kwargs, train_kwargs in configs:
        # Skip already completed experiments
        history_path = OUTPUT_DIR / exp_name / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            best_acc = max(history["val_acc"])
            results[exp_name] = best_acc
            print(f"\n>>> Skipping {exp_name} (already done, best val acc: {best_acc:.4f})")
            flush()
            continue

        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp_name}")
        print(f"{'#'*60}")
        flush()

        # Reuse dataloaders across arch experiments (same img_size + augmentation)
        if shared_loaders is None:
            train_loader, val_loader = get_dataloaders(**data_kwargs)
            shared_loaders = (train_loader, val_loader)
        else:
            train_loader, val_loader = shared_loaders

        model = get_model(model_name)
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            scheduler_type="cosine",
            **train_kwargs,
        )
        history = trainer.train()
        flush()

        # Save plots
        plot_training_curves(history, trainer.exp_dir / "training_curves.png")
        cm = trainer.get_confusion_matrix()
        plot_confusion_matrix(cm, trainer.exp_dir / "confusion_matrix.png")
        per_class = trainer.get_per_class_accuracy()
        plot_per_class_accuracy(per_class, trainer.exp_dir / "per_class_accuracy.png", train_counts)

        results[exp_name] = trainer.best_val_acc

        # Free GPU memory before next experiment (but keep dataloaders)
        del model, trainer
        mps_cleanup()
        flush()

    # Comparison plot
    plot_experiment_comparison(results, OUTPUT_DIR / "architecture_comparison.png")
    return results


def run_augmentation_ablation():
    """Experiment 5: Compare augmentation levels on best architecture."""
    results = {}

    for aug_level in ["none", "basic", "medium", "heavy"]:
        exp_name = f"5_aug_{aug_level}"

        # Skip completed
        history_path = OUTPUT_DIR / exp_name / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                h = json.load(f)
            results[aug_level] = max(h["val_acc"])
            print(f"\n>>> Skipping {exp_name} (already done, best val acc: {results[aug_level]:.4f})")
            flush()
            continue

        print(f"\n{'#'*60}")
        print(f"# Augmentation ablation: {aug_level}")
        print(f"{'#'*60}")
        flush()

        # Each aug level needs different transforms, so separate dataloaders
        train_loader, val_loader = get_dataloaders(
            img_size=128, augmentation=aug_level,
        )
        model = get_model("resnet")
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            epochs=20, lr=1e-3,
            scheduler_type="cosine",
        )
        history = trainer.train()
        flush()
        plot_training_curves(history, trainer.exp_dir / "training_curves.png")
        results[aug_level] = trainer.best_val_acc
        del model, trainer, train_loader, val_loader
        mps_cleanup()
        flush()

    plot_experiment_comparison(results, OUTPUT_DIR / "augmentation_comparison.png")
    return results


def run_lr_schedule_comparison():
    """Experiment 6: Compare LR schedules."""
    results = {}

    # Load data once for all LR experiments (all use medium augmentation)
    shared_loaders = None

    for sched in ["step", "cosine", "onecycle"]:
        exp_name = f"6_lr_{sched}"

        # Skip completed
        history_path = OUTPUT_DIR / exp_name / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                h = json.load(f)
            results[sched] = max(h["val_acc"])
            print(f"\n>>> Skipping {exp_name} (already done, best val acc: {results[sched]:.4f})")
            flush()
            continue

        print(f"\n{'#'*60}")
        print(f"# LR schedule: {sched}")
        print(f"{'#'*60}")
        flush()

        if shared_loaders is None:
            train_loader, val_loader = get_dataloaders(
                img_size=128, augmentation="medium",
            )
            shared_loaders = (train_loader, val_loader)
        else:
            train_loader, val_loader = shared_loaders

        model = get_model("resnet")
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            epochs=20, lr=1e-3,
            scheduler_type=sched,
        )
        history = trainer.train()
        flush()
        plot_training_curves(history, trainer.exp_dir / "training_curves.png")
        results[sched] = trainer.best_val_acc
        del model, trainer
        mps_cleanup()
        flush()

    plot_experiment_comparison(results, OUTPUT_DIR / "lr_schedule_comparison.png")
    return results


def run_label_smoothing_experiment():
    """Experiment 7: Label smoothing effect."""
    results = {}

    # Load data once (all use medium augmentation)
    shared_loaders = None

    for ls in [0.0, 0.05, 0.1, 0.2]:
        exp_name = f"7_ls_{ls}"

        # Skip completed
        history_path = OUTPUT_DIR / exp_name / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                h = json.load(f)
            results[f"ls={ls}"] = max(h["val_acc"])
            print(f"\n>>> Skipping {exp_name} (already done, best val acc: {results[f'ls={ls}']:.4f})")
            flush()
            continue

        print(f"\n{'#'*60}")
        print(f"# Label smoothing: {ls}")
        print(f"{'#'*60}")
        flush()

        if shared_loaders is None:
            train_loader, val_loader = get_dataloaders(
                img_size=128, augmentation="medium",
            )
            shared_loaders = (train_loader, val_loader)
        else:
            train_loader, val_loader = shared_loaders

        model = get_model("resnet")
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            epochs=20, lr=1e-3,
            scheduler_type="cosine",
            label_smoothing=ls,
        )
        history = trainer.train()
        flush()
        plot_training_curves(history, trainer.exp_dir / "training_curves.png")
        results[f"ls={ls}"] = trainer.best_val_acc
        del model, trainer
        mps_cleanup()
        flush()

    plot_experiment_comparison(results, OUTPUT_DIR / "label_smoothing_comparison.png")
    return results


def run_final_model():
    """Train the best configuration at higher resolution and generate submission."""
    exp_name = "8_final"
    print(f"\n{'#'*60}")
    print(f"# FINAL MODEL — Higher Resolution Training")
    print(f"{'#'*60}")

    train_loader, val_loader = get_dataloaders(
        img_size=160, augmentation="heavy", batch_size=48,
    )
    model = get_model("attention_resnet")
    trainer = Trainer(
        model, train_loader, val_loader,
        experiment_name=exp_name,
        epochs=30, lr=1e-3,
        scheduler_type="onecycle",
        label_smoothing=0.1,
    )
    history = trainer.train()
    plot_training_curves(history, trainer.exp_dir / "training_curves.png")

    # Load best model for analysis and submission
    model.load_state_dict(torch.load(trainer.exp_dir / "best_model.pt", map_location=DEVICE))
    trainer.model = model

    # Confusion matrix & per-class accuracy
    cm = trainer.get_confusion_matrix()
    plot_confusion_matrix(cm, trainer.exp_dir / "confusion_matrix.png")

    _, all_labels = load_train_data()
    train_counts = dict(Counter([l - 1 for l in all_labels]))
    per_class = trainer.get_per_class_accuracy()
    plot_per_class_accuracy(per_class, trainer.exp_dir / "per_class_accuracy.png", train_counts)

    # Grad-CAM on last conv layer
    if hasattr(model, "layer4"):
        target_layer = model.layer4[-1].conv2
        val_imgs = []
        for imgs, _ in val_loader:
            val_imgs = imgs[:8]
            break
        plot_gradcam(val_imgs, model, target_layer, trainer.exp_dir / "gradcam.png",
                     DATASET_MEAN, DATASET_STD)

    # t-SNE
    features, labels = compute_features(model, val_loader)
    plot_tsne(features, labels, trainer.exp_dir / "tsne.png")

    # Submission
    test_loader, test_names = get_test_loader(img_size=160, batch_size=48)
    trainer.save_submission(test_loader, test_names, "submission_final.csv")

    return trainer.best_val_acc


def run_progressive_resizing():
    """Progressive resizing: train at 64 -> 128 -> 224, loading weights between stages.

    This acts as curriculum learning — the network learns coarse features fast
    on small images, then refines on larger ones. Combined with mixup and a
    wider model for more capacity.
    """
    exp_name = "9_progressive"
    exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        {"img_size": 64,  "epochs": 25, "lr": 2e-3, "batch_size": 128, "augmentation": "basic",  "mixup": 0.0},
        {"img_size": 128, "epochs": 25, "lr": 1e-3, "batch_size": 64,  "augmentation": "medium", "mixup": 0.2},
        {"img_size": 224, "epochs": 30, "lr": 5e-4, "batch_size": 32,  "augmentation": "medium", "mixup": 0.4},
    ]

    model = get_model("resnet_large")
    all_history = {"stages": []}

    for i, stage in enumerate(stages):
        stage_dir = exp_dir / f"stage{i+1}_{stage['img_size']}px"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Check if this stage was already completed
        history_path = stage_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            best_acc = max(history["val_acc"])
            print(f"\n>>> Skipping stage {i+1} ({stage['img_size']}px) — already done, best val acc: {best_acc:.4f}")
            # Load the best model from this stage
            model.load_state_dict(torch.load(stage_dir / "best_model.pt", map_location=DEVICE))
            all_history["stages"].append({"size": stage["img_size"], "best_acc": best_acc})
            flush()
            continue

        print(f"\n{'#'*60}")
        print(f"# Progressive Stage {i+1}/3: {stage['img_size']}x{stage['img_size']}")
        print(f"# Epochs: {stage['epochs']}, LR: {stage['lr']}, Batch: {stage['batch_size']}")
        print(f"# Augmentation: {stage['augmentation']}, Mixup alpha: {stage['mixup']}")
        print(f"{'#'*60}")
        flush()

        train_loader, val_loader = get_dataloaders(
            img_size=stage["img_size"],
            batch_size=stage["batch_size"],
            augmentation=stage["augmentation"],
        )

        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=f"{exp_name}/stage{i+1}_{stage['img_size']}px",
            epochs=stage["epochs"],
            lr=stage["lr"],
            scheduler_type="onecycle",
            label_smoothing=0.1,
            mixup_alpha=stage["mixup"],
        )
        history = trainer.train()
        flush()

        plot_training_curves(history, stage_dir / "training_curves.png")

        # Load best weights for next stage
        model.load_state_dict(torch.load(stage_dir / "best_model.pt", map_location=DEVICE))
        best_acc = trainer.best_val_acc
        all_history["stages"].append({"size": stage["img_size"], "best_acc": best_acc})

        del trainer, train_loader, val_loader
        mps_cleanup()
        flush()

    # Final evaluation at 224px
    print(f"\n{'#'*60}")
    print(f"# Final evaluation & submission at 224x224")
    print(f"{'#'*60}")
    flush()

    # Reload best model from last stage
    last_stage_dir = exp_dir / f"stage3_224px"
    model.load_state_dict(torch.load(last_stage_dir / "best_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Confusion matrix & per-class accuracy
    train_loader, val_loader = get_dataloaders(img_size=224, batch_size=32, augmentation="basic")
    trainer = Trainer(model, train_loader, val_loader, experiment_name=exp_name, epochs=1)
    cm = trainer.get_confusion_matrix()
    plot_confusion_matrix(cm, exp_dir / "confusion_matrix.png")

    _, all_labels = load_train_data()
    train_counts = dict(Counter([l - 1 for l in all_labels]))
    per_class = trainer.get_per_class_accuracy()
    plot_per_class_accuracy(per_class, exp_dir / "per_class_accuracy.png", train_counts)

    # Grad-CAM
    if hasattr(model, "layer4"):
        target_layer = model.layer4[-1].conv2
        for imgs, _ in val_loader:
            val_imgs = imgs[:8]
            break
        plot_gradcam(val_imgs, model, target_layer, exp_dir / "gradcam.png",
                     DATASET_MEAN, DATASET_STD)

    # t-SNE
    features, labels = compute_features(model, val_loader)
    plot_tsne(features, labels, exp_dir / "tsne.png")

    # Submission
    test_loader, test_names = get_test_loader(img_size=224, batch_size=32)
    trainer.save_submission(test_loader, test_names, "submission_progressive.csv")

    # Save combined history
    with open(exp_dir / "all_stages.json", "w") as f:
        json.dump(all_history, f, indent=2)

    final_acc = all_history["stages"][-1]["best_acc"]
    print(f"\nProgressive resizing complete! Final best val acc: {final_acc:.4f}")
    return final_acc


def run_analysis_only():
    """Run analysis on already-trained models."""
    all_results = {}

    for exp_dir in sorted(OUTPUT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        history_path = exp_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            best_acc = max(history["val_acc"])
            all_results[exp_dir.name] = best_acc
            print(f"{exp_dir.name}: best val acc = {best_acc:.4f}")

    if all_results:
        plot_experiment_comparison(all_results, OUTPUT_DIR / "all_experiments_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Food Recognition Experiments")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "baseline", "architecture", "augmentation",
                                 "lr_schedule", "label_smoothing", "final",
                                 "progressive", "analyze"],
                        help="Which experiment to run")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.exp == "all":
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print("="*60)

        arch_results = run_architecture_experiments()
        aug_results = run_augmentation_ablation()
        lr_results = run_lr_schedule_comparison()
        ls_results = run_label_smoothing_experiment()
        final_acc = run_final_model()

        # Final summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print("\nArchitecture comparison:")
        for k, v in arch_results.items():
            print(f"  {k}: {v:.4f}")
        print("\nAugmentation ablation:")
        for k, v in aug_results.items():
            print(f"  {k}: {v:.4f}")
        print("\nLR schedule comparison:")
        for k, v in lr_results.items():
            print(f"  {k}: {v:.4f}")
        print("\nLabel smoothing:")
        for k, v in ls_results.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nFinal model: {final_acc:.4f}")

        # Save summary
        all_results = {**arch_results, **{f"aug_{k}": v for k, v in aug_results.items()},
                       **{f"lr_{k}": v for k, v in lr_results.items()}, **ls_results,
                       "final": final_acc}
        plot_experiment_comparison(all_results, OUTPUT_DIR / "all_experiments_comparison.png")

        with open(OUTPUT_DIR / "experiment_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    elif args.exp == "baseline":
        train_loader, val_loader = get_dataloaders(augmentation="basic")
        model = get_model("baseline")
        trainer = Trainer(model, train_loader, val_loader, experiment_name="1_baseline", epochs=30)
        trainer.train()
        plot_training_curves(trainer.history, trainer.exp_dir / "training_curves.png")

    elif args.exp == "architecture":
        run_architecture_experiments()

    elif args.exp == "augmentation":
        run_augmentation_ablation()

    elif args.exp == "lr_schedule":
        run_lr_schedule_comparison()

    elif args.exp == "label_smoothing":
        run_label_smoothing_experiment()

    elif args.exp == "final":
        run_final_model()

    elif args.exp == "progressive":
        run_progressive_resizing()

    elif args.exp == "analyze":
        run_analysis_only()


if __name__ == "__main__":
    main()
