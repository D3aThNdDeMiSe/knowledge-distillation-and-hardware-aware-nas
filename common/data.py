"""
Data pipeline for CIFAR-10 / CIFAR-100.

Windows + PyTorch DataLoader workers are slow (process spawning, pickling), and
CIFAR is tiny (50k x 32x32x3 uint8 = 150 MB), so the whole dataset is kept on
the GPU as a uint8 tensor and augmentation (random-crop with padding +
horizontal flip) is done on the GPU in batch. This removes the input-pipeline
bottleneck completely and makes epochs several times faster.

The same loader also supports a target `resolution` (used by the NAS search
space, which treats input resolution as a searchable parameter).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

# Per-channel statistics of the training sets (computed once, standard values).
CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# Hugging Face mirror of the official CIFAR datasets (uoft-cs). Same images/labels as
# the Toronto tarballs, but downloads ~50x faster. Each file is decoded once into a
# compact .npz cache (data/<name>_<split>.npz) that all scripts read afterwards.
HF_URLS = {
    ("cifar10", "train"): "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    ("cifar10", "test"): "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
    ("cifar100", "train"): "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
    ("cifar100", "test"): "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
}
HF_LOCAL = {("cifar10", "train"): "c10_train", ("cifar10", "test"): "c10_test",
            ("cifar100", "train"): "c100_train", ("cifar100", "test"): "c100_test"}


def _build_cache(name, split, root):
    """Download (if needed) the parquet file and decode the PNG bytes into a uint8 array."""
    import io
    import json
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    pq_path = os.path.join(root, "hf", HF_LOCAL[(name, split)] + ".parquet")
    if not os.path.exists(pq_path):
        os.makedirs(os.path.dirname(pq_path), exist_ok=True)
        print(f"Downloading {name}/{split} -> {pq_path}")
        urllib.request.urlretrieve(HF_URLS[(name, split)], pq_path)
    table = pq.read_table(pq_path)
    label_col = "label" if name == "cifar10" else "fine_label"
    imgs = table.column("img").to_pylist()
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in imgs])
    y = np.asarray(table.column(label_col).to_pylist(), dtype=np.int64)
    meta = json.loads(table.schema.metadata[b"huggingface"])
    names = meta["info"]["features"][label_col]["names"]
    out = os.path.join(root, f"{name}_{split}.npz")
    np.savez(out, x=x.astype(np.uint8), y=y, classes=np.array(names))
    return out


def load_cifar_numpy(name, train, root=DATA_ROOT):
    """Return (images uint8 [N,32,32,3], labels int64 [N])."""
    split = "train" if train else "test"
    path = os.path.join(root, f"{name}_{split}.npz")
    if not os.path.exists(path):
        _build_cache(name, split, root)
    d = np.load(path)
    return d["x"], d["y"]


def class_names(name, root=DATA_ROOT):
    path = os.path.join(root, f"{name}_test.npz")
    if not os.path.exists(path):
        _build_cache(name, "test", root)
    return [str(c) for c in np.load(path)["classes"]]


def normalize(x, name):
    """x: float tensor in [0,1], shape [B,3,H,W] -> normalized with dataset statistics."""
    mean, std = CIFAR_STATS[name]
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class GPULoader:
    """Iterates over an in-GPU-memory dataset in mini-batches with on-GPU augmentation.

    Args:
        images: uint8 numpy array [N,32,32,3]
        labels: int64 numpy array [N]
        dataset: 'cifar10' | 'cifar100' (selects normalization statistics)
        batch_size: mini-batch size
        train: if True -> shuffle + random crop (pad 4) + horizontal flip
        resolution: output spatial size (images are bilinearly resized if != 32)
        drop_last: drop the last incomplete batch (training only)
    """

    def __init__(self, images, labels, dataset, batch_size=128, train=True,
                 device="cuda", resolution=32, drop_last=None, pad=4, channels_last=False):
        self.x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.from_numpy(labels).to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.resolution = resolution
        self.pad = pad
        self.drop_last = train if drop_last is None else drop_last
        # channels_last (NHWC) is faster for dense convs (ResNet-50 teacher), while the
        # depthwise-heavy students run faster in the default NCHW layout (measured).
        self.memory_format = torch.channels_last if channels_last else torch.contiguous_format

    def __len__(self):
        n = self.x.shape[0]
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    @property
    def num_samples(self):
        return self.x.shape[0]

    def _augment(self, x):
        """Random crop (with reflection padding) + random horizontal flip, per sample, on GPU."""
        b, c, h, w = x.shape
        p = self.pad
        xp = F.pad(x, (p, p, p, p), mode="reflect")
        # Random top-left offsets for every image in the batch.
        oy = torch.randint(0, 2 * p + 1, (b,), device=x.device)
        ox = torch.randint(0, 2 * p + 1, (b,), device=x.device)
        rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])  # [B,H]
        cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])  # [B,W]
        bi = torch.arange(b, device=x.device)[:, None, None]
        # Advanced indexing gathers a different crop window per sample -> [B,H,W,C]
        x = xp.permute(0, 2, 3, 1)[bi, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
        # Horizontal flip with probability 0.5.
        flip = torch.rand(b, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x

    def __iter__(self):
        n = self.x.shape[0]
        order = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.x[idx].float().div_(255.0)
            if self.train:
                x = self._augment(x)
            x = normalize(x, self.dataset)
            if self.resolution != x.shape[-1]:
                x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear",
                                  align_corners=False)
            yield x.contiguous(memory_format=self.memory_format), self.y[idx]


def get_loaders(dataset, batch_size=128, device="cuda", resolution=32, val_split=0, seed=0,
                eval_batch_size=512, channels_last=False):
    """Build train / (optional val) / test loaders.

    If val_split > 0, the last `val_split` images of a seeded permutation of the
    training set are held out as a validation set (used by NAS so that the test
    set is never touched during search).
    """
    xtr, ytr = load_cifar_numpy(dataset, train=True)
    xte, yte = load_cifar_numpy(dataset, train=False)
    val_loader = None
    if val_split > 0:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(xtr))
        val_idx, tr_idx = perm[:val_split], perm[val_split:]
        val_loader = GPULoader(xtr[val_idx], ytr[val_idx], dataset, eval_batch_size, train=False,
                               device=device, resolution=resolution, channels_last=channels_last)
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    train_loader = GPULoader(xtr, ytr, dataset, batch_size, train=True, device=device,
                             resolution=resolution, channels_last=channels_last)
    test_loader = GPULoader(xte, yte, dataset, eval_batch_size, train=False, device=device,
                            resolution=resolution, channels_last=channels_last)
    return train_loader, val_loader, test_loader
