"""
Train the ResNet-50 teacher on CIFAR-100 (Deliverable 1).

Example:
    python -m q1_kd.train_teacher --epochs 60
Outputs:
    checkpoints/teacher_resnet50.pt          (fp16 weights + metadata)
    results/q1/teacher_resnet50_history.csv  (per-epoch log)
"""
import argparse
import os
import time

import torch

from common.data import get_loaders
from common.utils import (CKPT_DIR, RESULTS_DIR, ce_loss_fn, count_params, evaluate, fit,
                          get_device, save_checkpoint, set_seed, write_csv)
from q1_kd.models import build_model


def main():
    p = argparse.ArgumentParser(description="Train ResNet-50 teacher on CIFAR-100")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(CKPT_DIR, "teacher_resnet50.pt"))
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_loader, _, test_loader = get_loaders("cifar100", args.batch_size, device,
                                               channels_last=True)
    model = build_model("resnet50", 100).to(device).to(memory_format=torch.channels_last)
    print(f"Teacher ResNet-50: {count_params(model) / 1e6:.2f}M params | device={device}")

    t0 = time.time()
    history = fit(model, train_loader, test_loader, device, args.epochs, args.lr, args.wd,
                  ce_loss_fn(), "teacher", warmup_epochs=2)
    train_time = time.time() - t0
    top1, top5, _ = evaluate(model, test_loader, device)
    print(f"Teacher final: top1={top1:.2f}% top5={top5:.2f}% | {train_time / 60:.1f} min")

    save_checkpoint(args.out, model, {"arch": "resnet50", "num_classes": 100, "dataset": "cifar100",
                                      "top1": top1, "top5": top5, "epochs": args.epochs,
                                      "train_time_s": train_time, "args": vars(args)}, half=True)
    write_csv(os.path.join(RESULTS_DIR, "q1", "teacher_resnet50_history.csv"), history)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
