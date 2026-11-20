"""
Train the best discovered architecture FROM SCRATCH (Deliverable 8).

Reads results/q2/best_architecture.json (written by q2_nas.search), re-initialises the
network (no weights are re-used from the search), trains on the full 50k CIFAR-10
training set and reports test accuracy.

Example:
    python -m q2_nas.train_best --epochs 60
Outputs:
    checkpoints/nas_best.pt                    (weights + architecture config)
    results/q2/nas_best_history.csv
"""
import argparse
import os
import time

from common.data import get_loaders
from common.utils import (CKPT_DIR, RESULTS_DIR, ce_loss_fn, count_params, evaluate, fit,
                          get_device, read_json, save_checkpoint, set_seed, write_csv)
from q2_nas.generator import build_from_config
from q2_nas.search_space import describe


def main():
    p = argparse.ArgumentParser(description="Train the best NAS architecture from scratch")
    p.add_argument("--arch", default=os.path.join(RESULTS_DIR, "q2", "best_architecture.json"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    best = read_json(args.arch)
    cfg = best["config"]
    model = build_from_config(cfg, num_classes=10).to(device)
    print(f"Best architecture (trial {best['best_trial']}): {describe(cfg)}")
    print(f"{count_params(model) / 1e6:.3f}M params, input {cfg['resolution']}x{cfg['resolution']}")

    train_loader, _, test_loader = get_loaders("cifar10", args.batch_size, device,
                                               resolution=cfg["resolution"])
    t0 = time.time()
    history = fit(model, train_loader, test_loader, device, args.epochs, args.lr, args.wd,
                  ce_loss_fn(), "nas_best", warmup_epochs=2)
    train_time = time.time() - t0
    top1, top5, _ = evaluate(model, test_loader, device)
    path = os.path.join(CKPT_DIR, "nas_best.pt")
    save_checkpoint(path, model, {"arch": "nas", "config": cfg, "num_classes": 10,
                                  "dataset": "cifar10", "top1": top1, "top5": top5,
                                  "epochs": args.epochs, "train_time_s": train_time,
                                  "args": vars(args)})
    write_csv(os.path.join(RESULTS_DIR, "q2", "nas_best_history.csv"), history)
    print(f"NAS best (from scratch): test top1={top1:.2f}% top5={top5:.2f}% | "
          f"{train_time / 60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
