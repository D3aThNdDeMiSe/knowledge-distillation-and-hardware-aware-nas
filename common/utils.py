"""
Generic training helpers: seeding, schedulers, train/eval loops, checkpoint I/O,
CSV/JSON logging. Used by teacher, baseline, KD, and NAS training scripts.
"""
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
ONNX_DIR = os.path.join(ROOT, "onnx_models")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return "cuda"
    return "cpu"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_optimizer(model, lr, weight_decay, momentum=0.9):
    """SGD + Nesterov. Weight decay is not applied to BatchNorm params and biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = all(p.is_cuda for g in groups for p in g["params"])  # single fused CUDA kernel
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True,
                           fused=fused or None)


def warmup_cosine(optimizer, total_steps, warmup_steps):
    """Linear warm-up followed by cosine decay to zero (stepped every iteration)."""

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def autocast(device):
    """bfloat16 autocast on GPU (no GradScaler needed); no-op on CPU."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (top1 %, top5 %, mean CE loss) of `model` on `loader`."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct1 = correct5 = 0
    total_loss, n = 0.0, 0
    for x, y in loader:
        with autocast(device):
            out = model(x)
        out = out.float()
        total_loss += ce(out, y).item()
        top5 = out.topk(min(5, out.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return 100.0 * correct1 / n, 100.0 * correct5 / n, total_loss / n


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn, max_grad_norm=None):
    """Generic epoch. `loss_fn(model, x, y) -> (loss, logits, extra_dict)` lets
    the same loop serve plain CE training and knowledge distillation."""
    model.train()
    # Running sums are kept as GPU tensors and only synchronised once per epoch:
    # calling .item() every iteration would stall the CPU until the GPU finishes,
    # destroying CPU/GPU overlap (very noticeable for small, launch-bound students).
    sums, n = {}, 0
    for x, y in loader:
        with autocast(device):
            loss, logits, extra = loss_fn(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        bs = y.numel()
        n += bs
        vals = {"loss": loss.detach().float(),
                "acc": (logits.detach().argmax(1) == y).float().mean() * 100, **extra}
        for k, v in vals.items():
            v = v.detach().float() * bs if torch.is_tensor(v) else float(v) * bs
            sums[k] = sums[k] + v if k in sums else v
    return {k: float(v) / max(1, n) for k, v in sums.items()}


def ce_loss_fn(label_smoothing=0.0):
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def fn(model, x, y):
        logits = model(x)
        return ce(logits.float(), y), logits, {}

    return fn


def fit(model, train_loader, test_loader, device, epochs, lr, weight_decay, loss_fn,
        log_name, warmup_epochs=1, extra_params=None, print_every=1, max_grad_norm=None):
    """Full training run with warm-up + cosine LR. Returns history (list of dicts).

    `extra_params` (e.g. KD feature regressors) are optimised jointly with the model.
    """
    params_model = model
    if extra_params is not None:
        params_model = nn.ModuleList([model, extra_params])
    optimizer = build_optimizer(params_model, lr, weight_decay)
    steps = epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, steps, warmup_epochs * len(train_loader))
    history = []
    best = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_fn,
                             max_grad_norm=max_grad_norm)
        top1, top5, te_loss = evaluate(model, test_loader, device)
        best = max(best, top1)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": round(v, 4) for k, v in tr.items()},
               "test_loss": round(te_loss, 4), "test_top1": round(top1, 2),
               "test_top5": round(top5, 2), "epoch_time_s": round(time.time() - t0, 1)}
        history.append(row)
        if epoch % print_every == 0 or epoch == epochs:
            extras = " ".join(f"{k}={v:.3f}" for k, v in tr.items() if k not in ("loss", "acc"))
            print(f"[{log_name}] ep {epoch:3d}/{epochs} | loss {tr['loss']:.3f} "
                  f"train_acc {tr['acc']:.2f} | test top1 {top1:.2f} top5 {top5:.2f} "
                  f"(best {best:.2f}) {extras} | {row['epoch_time_s']}s", flush=True)
    return history


def save_checkpoint(path, model, meta, half=False):
    """Save weights + metadata. half=True stores weights in fp16 (halves file size;
    used for the 23.7M-param teacher so it stays well under GitHub's 100 MB limit)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {k: (v.half() if half and v.is_floating_point() else v).cpu()
          for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, **meta}, path)


def load_state(model, ckpt):
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    return model


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)
