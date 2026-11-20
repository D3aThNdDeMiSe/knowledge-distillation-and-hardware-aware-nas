"""
Baseline training pipeline for the three students WITHOUT distillation (Deliverable 3).
Students are trained with plain cross-entropy on the hard labels, using exactly the
same data pipeline, optimiser, schedule and epochs as the distilled students, so the
comparison isolates the effect of knowledge distillation.

Example:
    python -m q1_kd.train_baseline --models mobilenetv2 squeezenet efficientnet_lite0 --epochs 60
Outputs (per student):
    checkpoints/baseline_<student>.pt
    results/q1/baseline_<student>_history.csv
"""
import argparse
import os
import time

import torch

from common.data import get_loaders
from common.utils import (CKPT_DIR, RESULTS_DIR, ce_loss_fn, count_params, evaluate, fit,
                          get_device, save_checkpoint, set_seed, write_csv)
from q1_kd.models import STUDENTS, build_model


def train_baseline(name, args, device, loaders):
    set_seed(args.seed)
    train_loader, test_loader = loaders
    model = build_model(name, 100).to(device)
    print(f"\n=== Baseline {name}: {count_params(model) / 1e6:.2f}M params ===")
    t0 = time.time()
    history = fit(model, train_loader, test_loader, device, args.epochs, args.lr, args.wd,
                  ce_loss_fn(), f"baseline/{name}", warmup_epochs=2)
    train_time = time.time() - t0
    top1, top5, _ = evaluate(model, test_loader, device)
    path = os.path.join(CKPT_DIR, f"baseline_{name}.pt")
    save_checkpoint(path, model, {"arch": name, "num_classes": 100, "dataset": "cifar100",
                                  "mode": "baseline", "top1": top1, "top5": top5,
                                  "epochs": args.epochs, "train_time_s": train_time,
                                  "args": vars(args)})
    write_csv(os.path.join(RESULTS_DIR, "q1", f"baseline_{name}_history.csv"), history)
    print(f"Baseline {name}: top1={top1:.2f}% top5={top5:.2f}% -> {path}")
    return top1


def main():
    p = argparse.ArgumentParser(description="Train student baselines (no KD) on CIFAR-100")
    p.add_argument("--models", nargs="+", default=STUDENTS, choices=STUDENTS)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    train_loader, _, test_loader = get_loaders("cifar100", args.batch_size, device)
    for name in args.models:
        train_baseline(name, args, device, (train_loader, test_loader))


if __name__ == "__main__":
    main()
