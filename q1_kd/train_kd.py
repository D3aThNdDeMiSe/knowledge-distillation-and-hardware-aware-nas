"""
Train the three students WITH knowledge distillation from the ResNet-50 teacher
(Deliverable 4). Uses the same optimiser / schedule / epochs as the baselines.

Example:
    python -m q1_kd.train_kd --models mobilenetv2 squeezenet efficientnet_lite0 --epochs 60
    # ablation: soft targets only, no feature distillation
    python -m q1_kd.train_kd --models mobilenetv2 --feat-layers none --tag kd_logits
Outputs (per student):
    checkpoints/kd_<student>.pt                (student weights only; regressors are training-only)
    results/q1/kd_<student>_history.csv
"""
import argparse
import os
import time

import torch

from common.data import get_loaders
from common.utils import (CKPT_DIR, RESULTS_DIR, count_params, evaluate, fit, get_device,
                          load_state, save_checkpoint, set_seed, write_csv)
from q1_kd.kd import DistillationLoss, make_kd_loss_fn
from q1_kd.models import STUDENTS, build_model


def load_teacher(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    teacher = load_state(build_model(ckpt["arch"], ckpt["num_classes"]), ckpt)
    teacher = teacher.to(device).to(memory_format=torch.channels_last).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, ckpt


def train_kd(name, args, device, teacher, loaders):
    set_seed(args.seed)
    train_loader, test_loader = loaders
    student = build_model(name, 100).to(device)
    feat_layers = [] if args.feat_layers == ["none"] else args.feat_layers
    criterion = DistillationLoss(student.feature_channels, teacher.feature_channels, feat_layers,
                                 T=args.T, alpha=args.alpha, beta=args.beta,
                                 use_hard=not args.no_hard).to(device)
    print(f"\n=== KD {name} ({args.tag}): {count_params(student) / 1e6:.2f}M params | "
          f"T={args.T} alpha={args.alpha} beta={args.beta} feat={feat_layers} "
          f"hard={not args.no_hard} ===")
    t0 = time.time()
    history = fit(student, train_loader, test_loader, device, args.epochs, args.lr, args.wd,
                  make_kd_loss_fn(teacher, criterion), f"{args.tag}/{name}", warmup_epochs=2,
                  extra_params=criterion.regressors)
    train_time = time.time() - t0
    top1, top5, _ = evaluate(student, test_loader, device)
    path = os.path.join(CKPT_DIR, f"{args.tag}_{name}.pt")
    save_checkpoint(path, student, {"arch": name, "num_classes": 100, "dataset": "cifar100",
                                    "mode": args.tag, "top1": top1, "top5": top5,
                                    "epochs": args.epochs, "train_time_s": train_time,
                                    "kd": {"T": args.T, "alpha": args.alpha, "beta": args.beta,
                                           "feat_layers": feat_layers, "hard": not args.no_hard},
                                    "args": vars(args)})
    write_csv(os.path.join(RESULTS_DIR, "q1", f"{args.tag}_{name}_history.csv"), history)
    print(f"KD {name}: top1={top1:.2f}% top5={top5:.2f}% -> {path}")
    return top1


def main():
    p = argparse.ArgumentParser(description="Knowledge distillation: ResNet-50 -> students")
    p.add_argument("--models", nargs="+", default=STUDENTS, choices=STUDENTS)
    p.add_argument("--teacher", default=os.path.join(CKPT_DIR, "teacher_resnet50.pt"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    p.add_argument("--alpha", type=float, default=0.9, help="soft-target weight (hard = 1-alpha)")
    p.add_argument("--beta", type=float, default=2.0, help="feature-distillation weight per layer")
    p.add_argument("--feat-layers", nargs="+", default=["mid", "final"],
                   help="feature taps to distil: mid final | none")
    p.add_argument("--no-hard", action="store_true", help="disable the hard-label CE term")
    p.add_argument("--tag", default="kd", help="checkpoint/result prefix (e.g. kd, kd_logits)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    teacher, tck = load_teacher(args.teacher, device)
    train_loader, _, test_loader = get_loaders("cifar100", args.batch_size, device)
    t1, _, _ = evaluate(teacher, test_loader, device)
    print(f"Loaded teacher {tck['arch']} (test top1 {t1:.2f}%)")
    for name in args.models:
        train_kd(name, args, device, teacher, (train_loader, test_loader))


if __name__ == "__main__":
    main()
