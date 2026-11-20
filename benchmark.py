"""
Automated benchmarking of every trained model (Deliverable 11).

For each model (teacher, 3 baseline students, 3 distilled students, NAS model):
  * #parameters, MACs (counted with forward hooks), checkpoint and ONNX file size
  * test accuracy with PyTorch (GPU if available) and with ONNX Runtime on CPU
  * CPU latency, batch 1: PyTorch eager (1 thread) and ONNX Runtime (1 and 4 threads)
  * ONNX Runtime CPU throughput on the full test set (batch 256, all threads)
  * peak inference memory (ONNX Runtime, isolated subprocess)
Run it on an otherwise idle machine for stable latency numbers.

Example:
    python benchmark.py
Outputs:
    results/benchmark.csv                 (all models, all metrics)
    results/q1/kd_vs_baseline.csv         (accuracy gain from distillation per student)
"""
import argparse
import os
import platform

import numpy as np
import torch
import torch.nn as nn

from common.data import GPULoader, load_cifar_numpy
from common.utils import RESULTS_DIR, evaluate, get_device, write_csv
from common.zoo import ALL_MODEL_IDS, STUDENT_ARCHS, available_models, checkpoint_path, load_model, onnx_path
from infer import Predictor
from q2_nas.hardware import measure_latency_ort, measure_latency_torch, measure_peak_memory


def count_macs(model, resolution):
    """Multiply-accumulate operations of one forward pass (conv + linear layers)."""
    macs = [0]

    def conv_hook(m, inp, out):
        macs[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * \
            m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(torch.randn(1, 3, resolution, resolution))
    for h in hooks:
        h.remove()
    return macs[0]


def group_of(mid):
    return ("teacher" if mid.startswith("teacher") else "baseline" if mid.startswith("baseline")
            else "kd" if mid.startswith("kd") else "nas")


def main():
    p = argparse.ArgumentParser(description="Benchmark all trained models")
    p.add_argument("--models", nargs="+", default=ALL_MODEL_IDS)
    p.add_argument("--runs", type=int, default=200, help="timed runs per latency measurement")
    args = p.parse_args()

    device = get_device()
    models = available_models(args.models)
    test_loaders = {}
    rows = []
    print(f"CPU: {platform.processor()} | torch {torch.__version__} | device for accuracy: {device}")
    for mid in models:
        model, info = load_model(mid, "cpu")
        res, ds = info["resolution"], info["dataset"]
        onnx_file = onnx_path(mid)
        if not os.path.exists(onnx_file):
            raise SystemExit(f"{onnx_file} missing - run `python export_onnx.py` first")
        row = {"model": mid, "group": group_of(mid), "arch": info["arch"], "dataset": ds,
               "input": f"{res}x{res}",
               "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
               "MMACs": round(count_macs(model, res) / 1e6, 2),
               "ckpt_MB": round(os.path.getsize(checkpoint_path(mid)) / 2 ** 20, 3),
               "onnx_MB": round(os.path.getsize(onnx_file) / 2 ** 20, 3)}

        # --- accuracy: PyTorch (GPU) ---
        key = (ds, res)
        if key not in test_loaders:
            x, y = load_cifar_numpy(ds, train=False)
            test_loaders[key] = GPULoader(x, y, ds, 500, train=False, device=device, resolution=res)
        top1, top5, _ = evaluate(model.to(device), test_loaders[key], device)
        model.cpu()
        row.update({"top1_torch": round(top1, 2), "top5_torch": round(top5, 2)})

        # --- accuracy + throughput: ONNX Runtime CPU ---
        r = Predictor(mid, "onnx").evaluate()
        row.update({"top1_onnx": round(r["top1"], 2), "top5_onnx": round(r["top5"], 2),
                    "ort_throughput_img_s": round(r["img_per_s"], 1)})

        # --- latency (batch 1) ---
        row["torch_cpu_lat_ms_1t"] = round(measure_latency_torch(model, res, 1, runs=args.runs // 2)["median"], 3)
        for t in (1, 4):
            lat = measure_latency_ort(onnx_file, res, threads=t, runs=args.runs)
            row[f"ort_lat_ms_{t}t"] = round(lat["median"], 3)
            row[f"ort_lat_p95_ms_{t}t"] = round(lat["p95"], 3)

        # --- peak inference memory ---
        with open(onnx_file, "rb") as f:
            mem = measure_peak_memory(f.read(), res, threads=1)
        row["peak_mem_MB"] = None if mem["peak_memory_mb"] is None else round(mem["peak_memory_mb"], 2)
        rows.append(row)
        print(f"{mid:28s} {row['params_M']:7.3f}M {row['MMACs']:8.1f}MMAC  top1 torch {top1:6.2f} "
              f"onnx {row['top1_onnx']:6.2f} | ORT 1t {row['ort_lat_ms_1t']:7.3f}ms "
              f"4t {row['ort_lat_ms_4t']:7.3f}ms | torch 1t {row['torch_cpu_lat_ms_1t']:7.3f}ms | "
              f"{row['onnx_MB']:7.3f}MB | mem {row['peak_mem_MB']}MB", flush=True)

    write_csv(os.path.join(RESULTS_DIR, "benchmark.csv"), rows)

    by_id = {r["model"]: r for r in rows}
    teacher = by_id.get("teacher_resnet50")
    comp = []
    for a in STUDENT_ARCHS:
        b, k = by_id.get(f"baseline_{a}"), by_id.get(f"kd_{a}")
        if b and k:
            comp.append({"student": a, "params_M": b["params_M"], "MMACs": b["MMACs"],
                         "baseline_top1": b["top1_torch"], "kd_top1": k["top1_torch"],
                         "kd_gain_top1": round(k["top1_torch"] - b["top1_torch"], 2),
                         "baseline_top5": b["top5_torch"], "kd_top5": k["top5_torch"],
                         "teacher_top1": teacher["top1_torch"] if teacher else None,
                         "compression_vs_teacher": round(teacher["params_M"] / b["params_M"], 1) if teacher else None,
                         "speedup_vs_teacher_ort_1t": round(teacher["ort_lat_ms_1t"] / k["ort_lat_ms_1t"], 1) if teacher else None})
    if comp:
        write_csv(os.path.join(RESULTS_DIR, "q1", "kd_vs_baseline.csv"), comp)
        print("\nKD vs baseline (CIFAR-100 top-1):")
        for c in comp:
            print(f"  {c['student']:20s} baseline {c['baseline_top1']:6.2f}  KD {c['kd_top1']:6.2f}  "
                  f"gain {c['kd_gain_top1']:+.2f}")
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'benchmark.csv')}")


if __name__ == "__main__":
    main()
