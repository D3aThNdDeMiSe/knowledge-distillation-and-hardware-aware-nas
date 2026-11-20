"""
Export trained models to ONNX and verify them with ONNX Runtime (Deliverable 10).

For every model the exported graph is checked with onnx.checker and its outputs are
compared against PyTorch on random inputs (max |difference| must be tiny).

Example:
    python export_onnx.py                  # all student models + NAS model + teacher
    python export_onnx.py --models kd_mobilenetv2 nas_best
Outputs:
    onnx_models/<model_id>.onnx
    results/onnx_export.csv
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.utils import ONNX_DIR, RESULTS_DIR, read_json, write_csv, write_json
from common.zoo import ALL_MODEL_IDS, available_models, load_model, onnx_path


def export_one(model_id, opset=17):
    model, info = load_model(model_id, "cpu")
    res = info["resolution"]
    path = onnx_path(model_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dummy = torch.randn(1, 3, res, res)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(path))

    # Numerical verification: PyTorch vs ONNX Runtime on a random batch.
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(8, 3, res, res)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())
    agree = float((ref.argmax(1) == out.argmax(1)).mean())
    row = {"model": model_id, "onnx_file": os.path.relpath(path), "input_shape": f"Nx3x{res}x{res}",
           "opset": opset, "size_mb": round(os.path.getsize(path) / 2 ** 20, 3),
           "max_abs_diff_vs_torch": f"{max_diff:.2e}", "top1_agreement": agree}
    print(f"{model_id:30s} -> {row['onnx_file']:40s} {row['size_mb']:8.3f} MB  "
          f"max|diff|={max_diff:.2e}  argmax agreement={agree:.0%}")
    return row


def main():
    p = argparse.ArgumentParser(description="Export trained models to ONNX + verify with ORT")
    p.add_argument("--models", nargs="+", default=ALL_MODEL_IDS)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    models = available_models(args.models)
    missing = sorted(set(args.models) - set(models))
    if missing:
        print(f"(skipping models without a checkpoint: {missing})")
    rows = [export_one(m, args.opset) for m in models]
    write_csv(os.path.join(RESULTS_DIR, "onnx_export.csv"), rows)

    # Small JSON "model cards" so the ONNX inference path needs no PyTorch checkpoint.
    card_path = os.path.join(ONNX_DIR, "model_cards.json")
    cards = read_json(card_path) if os.path.exists(card_path) else {}
    for m in models:
        _, info = load_model(m, "cpu")
        cards[m] = {"dataset": info["dataset"], "resolution": info["resolution"],
                    "num_classes": info["num_classes"], "arch": info["arch"],
                    "config": info.get("config"), "pytorch_top1": round(info["top1"], 2)}
    write_json(card_path, cards)
    print(f"\nExported {len(rows)} models to {ONNX_DIR}")


if __name__ == "__main__":
    main()
