"""
Unified inference pipeline for ALL trained models (Deliverables 6-Q1, 9, 10).

One interface, two interchangeable back-ends:
  * --backend torch : PyTorch checkpoint (checkpoints/<id>.pt), CPU or CUDA
  * --backend onnx  : exported ONNX model (onnx_models/<id>.onnx) on CPU via ONNX Runtime
Pre-processing is pure NumPy/PIL and identical for both back-ends (resize -> scale ->
normalise with the training-set statistics -> model-specific input resolution), so the
ONNX path has no PyTorch dependency at all.

Supported models: teacher_resnet50, baseline_<student>, kd_<student> (CIFAR-100 students:
mobilenetv2, squeezenet, efficientnet_lite0) and nas_best (CIFAR-10 NAS model), or 'all'.

Examples:
    python infer.py --model kd_mobilenetv2 --images samples/cifar100      # classify images
    python infer.py --model nas_best --backend onnx --images samples/cifar10
    python infer.py --model all --backend onnx --eval                     # test-set accuracy
    python infer.py --model kd_efficientnet_lite0 --backend onnx --random 5
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
ALL_IDS = ["teacher_resnet50",
           "baseline_mobilenetv2", "baseline_squeezenet", "baseline_efficientnet_lite0",
           "kd_mobilenetv2", "kd_squeezenet", "kd_efficientnet_lite0", "nas_best"]


# ----------------------------------------------------------------------------- metadata
def model_card(model_id):
    """Dataset / input resolution of a model, read from the small JSON written at export
    time (so ONNX inference does not need PyTorch), falling back to the checkpoint."""
    card_path = os.path.join(ROOT, "onnx_models", "model_cards.json")
    if os.path.exists(card_path):
        with open(card_path) as f:
            cards = json.load(f)
        if model_id in cards:
            return cards[model_id]
    from common.zoo import load_meta
    meta = load_meta(model_id)
    return {"dataset": meta["dataset"],
            "resolution": meta.get("config", {}).get("resolution", 32),
            "num_classes": meta["num_classes"]}


def class_names(dataset):
    with open(os.path.join(ROOT, "assets", "class_names.json")) as f:
        return json.load(f)[dataset]


# ----------------------------------------------------------------------- preprocessing
def resize_bilinear(x, size):
    """Bilinear resize of a [N,C,H,W] float array, identical to
    torch.nn.functional.interpolate(mode='bilinear', align_corners=False)."""
    n, c, h, w = x.shape
    if (h, w) == (size, size):
        return x

    def coords(n_in, n_out):
        src = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        src = np.maximum(src, 0.0)
        i0 = np.floor(src).astype(np.int64)
        i0 = np.minimum(i0, n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (src - i0).astype(np.float32)

    y0, y1, wy = coords(h, size)
    x0, x1, wx = coords(w, size)
    top = x[:, :, y0][:, :, :, x0] * (1 - wx) + x[:, :, y0][:, :, :, x1] * wx
    bot = x[:, :, y1][:, :, :, x0] * (1 - wx) + x[:, :, y1][:, :, :, x1] * wx
    return (top * (1 - wy)[:, None] + bot * wy[:, None]).astype(np.float32)


def preprocess(images_uint8, dataset, resolution):
    """images: uint8 [N,32,32,3] -> normalised float32 [N,3,res,res]."""
    mean, std = (np.asarray(v, np.float32).reshape(1, 3, 1, 1) for v in STATS[dataset])
    x = images_uint8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    x = (x - mean) / std
    return resize_bilinear(x, resolution)


def load_image(path):
    """Any image file -> uint8 32x32x3 (CIFAR geometry)."""
    img = Image.open(path).convert("RGB")
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ------------------------------------------------------------------------------ backends
class OnnxBackend:
    def __init__(self, model_id, threads=0):
        import onnxruntime as ort
        path = os.path.join(ROOT, "onnx_models", f"{model_id}.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python export_onnx.py` first")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.name = "ONNX Runtime (CPU)"

    def __call__(self, x):
        return self.sess.run(None, {self.input: x})[0]


class TorchBackend:
    def __init__(self, model_id, device="cpu", threads=0):
        import torch
        from common.zoo import load_model
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.model, _ = load_model(model_id, device)
        self.name = f"PyTorch ({device})"

    def __call__(self, x):
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x).to(self.device)).float().cpu().numpy()


class Predictor:
    """Model-agnostic predictor: same API for every model and back-end."""

    def __init__(self, model_id, backend="onnx", device="cpu", threads=0):
        self.model_id = model_id
        card = model_card(model_id)
        self.dataset, self.resolution = card["dataset"], card["resolution"]
        self.classes = class_names(self.dataset)
        self.backend = (OnnxBackend(model_id, threads) if backend == "onnx"
                        else TorchBackend(model_id, device, threads))

    def logits(self, images_uint8, batch_size=256):
        outs = []
        for i in range(0, len(images_uint8), batch_size):
            x = preprocess(images_uint8[i:i + batch_size], self.dataset, self.resolution)
            outs.append(self.backend(np.ascontiguousarray(x)))
        return np.concatenate(outs)

    def predict(self, images_uint8, topk=3):
        z = self.logits(images_uint8)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        idx = np.argsort(-p, axis=1)[:, :topk]
        return [[(self.classes[j], float(p[i, j])) for j in row] for i, row in enumerate(idx)]

    def evaluate(self, data_dir=os.path.join(ROOT, "data")):
        d = np.load(os.path.join(data_dir, f"{self.dataset}_test.npz"))
        x, y = d["x"], d["y"]
        t0 = time.perf_counter()
        z = self.logits(x)
        dt = time.perf_counter() - t0
        top5 = np.argsort(-z, axis=1)[:, :5]
        return {"top1": float((top5[:, 0] == y).mean() * 100),
                "top5": float((top5 == y[:, None]).any(1).mean() * 100),
                "images": len(y), "seconds": dt, "img_per_s": len(y) / dt}


# ----------------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="Unified inference for all trained models")
    p.add_argument("--model", nargs="+", default=["kd_mobilenetv2"],
                   help=f"model id(s) or 'all': {ALL_IDS}")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--device", default="cpu", help="torch backend device (cpu/cuda)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = runtime default)")
    p.add_argument("--images", nargs="*", help="image files and/or folders to classify")
    p.add_argument("--random", type=int, default=0, help="classify N random test images")
    p.add_argument("--eval", action="store_true", help="evaluate on the full test set")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    ids = ALL_IDS if args.model == ["all"] else args.model
    files = []
    for item in args.images or []:
        files += sorted(glob.glob(os.path.join(item, "*.png")) + glob.glob(os.path.join(item, "*.jpg"))) \
            if os.path.isdir(item) else [item]

    if args.eval:
        print(f"{'model':30s} {'backend':22s} {'dataset':9s} {'res':>4s} {'top1':>7s} {'top5':>7s} {'img/s':>9s}")
        print("-" * 94)
    for mid in ids:
        pred = Predictor(mid, args.backend, args.device, args.threads)
        if args.eval:
            r = pred.evaluate()
            print(f"{mid:30s} {pred.backend.name:22s} {pred.dataset:9s} {pred.resolution:4d} "
                  f"{r['top1']:6.2f}% {r['top5']:6.2f}% {r['img_per_s']:9.1f}")
            continue
        imgs, names, truth = [], [], []
        if files:
            # Only images that belong to this model's dataset folder are used when the
            # sample folders are given (samples/cifar10 vs samples/cifar100).
            for f in files:
                folder = os.path.basename(os.path.dirname(os.path.abspath(f)))
                if folder in STATS and folder != pred.dataset:
                    continue
                imgs.append(load_image(f))
                names.append(os.path.basename(f))
                # sample files are named <index>_<true-class>.png
                stem = os.path.splitext(os.path.basename(f))[0]
                truth.append(stem.split("_", 1)[1] if "_" in stem else "?")
        if args.random:
            d = np.load(os.path.join(ROOT, "data", f"{pred.dataset}_test.npz"))
            sel = np.random.default_rng().choice(len(d["y"]), args.random, replace=False)
            for i in sel:
                imgs.append(d["x"][i])
                names.append(f"test[{i}]")
                truth.append(pred.classes[d["y"][i]])
        if not imgs:
            print("Nothing to do: pass --images, --random N or --eval")
            return
        t0 = time.perf_counter()
        results = pred.predict(np.stack(imgs), args.topk)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n=== {mid} | {pred.backend.name} | {pred.dataset} | input {pred.resolution}x"
              f"{pred.resolution} | {len(imgs)} images in {ms:.1f} ms ===")
        print(f"{'image':28s} {'true':16s} {'prediction':16s} {'conf':>6s}  top-{args.topk}")
        for n, t, r in zip(names, truth, results):
            mark = "OK " if r[0][0] == t else "   "
            others = ", ".join(f"{c} {pp:.2f}" for c, pp in r[1:])
            print(f"{n:28s} {t:16s} {r[0][0]:16s} {r[0][1]:6.2f}  {mark} {others}")


if __name__ == "__main__":
    main()
