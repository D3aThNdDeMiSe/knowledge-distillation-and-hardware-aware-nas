"""
Hardware-aware measurements (Deliverable 6) - everything here is MEASURED on the real
CPU, not estimated from FLOPs:

  * CPU inference latency  - ONNX Runtime (the deployment runtime), batch 1, a fixed
                              number of threads (default 1 -> emulates a single-core
                              edge CPU), warm-up runs then median / p95 of N runs.
                              PyTorch eager CPU latency is recorded as well.
  * model storage size     - size in bytes of the serialised ONNX model (the file that is
                              actually deployed) and of the PyTorch state_dict.
  * peak inference memory  - measured in a *fresh subprocess* that loads the ONNX model
                              into ONNX Runtime and runs inference; we record the process
                              peak working set (Windows) / max RSS (Linux) minus the
                              baseline after imports. "Where supported": returns None on
                              platforms without a peak-RSS API.

`check_constraints` implements the rejection rule used by the search.

CLI (used internally for the peak-memory subprocess, also handy for manual checks):
    python -m q2_nas.hardware --onnx onnx_models/nas_best.onnx --res 32
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# NOTE: torch is imported lazily inside the functions that need it, so the peak-memory
# subprocess (which only needs onnxruntime) starts fast and does not carry torch's RSS.

# Calibrated by profiling 25 random architectures from the space on this CPU
# (medians: 2.0 ms, 1.5 MB, 13 MB) -> roughly half of the space is infeasible.
DEFAULT_CONSTRAINTS = {
    "max_latency_ms": 2.0,   # ONNX Runtime, 1 CPU thread, batch 1
    "max_size_mb": 1.5,      # serialised fp32 ONNX model
    "max_memory_mb": 16.0,   # peak inference memory increase of the process
}


def export_onnx_bytes(model, resolution, opset=17):
    """Export a model to an in-memory ONNX file (fp32, dynamic batch)."""
    import torch

    model = model.eval().cpu()
    buf = io.BytesIO()
    dummy = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(model, dummy, buf, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    return buf.getvalue()


def state_dict_size_bytes(model):
    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def _stats(times_ms):
    t = np.asarray(times_ms)
    return {"mean": float(t.mean()), "median": float(np.median(t)),
            "p95": float(np.percentile(t, 95)), "std": float(t.std())}


def measure_latency_ort(onnx_model, resolution, threads=1, warmup=20, runs=100, batch=1):
    """Latency of ONNX Runtime CPU inference. `onnx_model` = path or bytes."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_model, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, 3, resolution, resolution).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return _stats(times)


def measure_latency_torch(model, resolution, threads=1, warmup=10, runs=50, batch=1):
    """Latency of PyTorch eager inference on CPU."""
    import torch

    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        model = model.eval().cpu()
        x = torch.randn(batch, 3, resolution, resolution)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(prev)
    return _stats(times)


def _process_peak_bytes():
    import psutil

    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):                    # Windows
        return mi.peak_wset
    try:                                            # Linux: ru_maxrss in KiB
        import resource
        if sys.platform.startswith("linux"):
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        pass
    return None


def _peak_memory_child(onnx_path, resolution, threads, runs=20):
    """Runs inside the fresh subprocess.

    The OS 'peak working set' counter cannot be reset and also contains the transient
    spike of importing onnxruntime, so the model's own peak is captured by a background
    thread that samples the resident set size (RSS) every ~0.5 ms while the session is
    created and inference runs. peak_memory_mb = max sampled RSS - RSS before loading.
    """
    import threading

    import onnxruntime as ort
    import psutil

    proc = psutil.Process()
    base_rss = proc.memory_info().rss
    peak = [base_rss]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, resolution, resolution).astype(np.float32)
    for _ in range(runs):
        sess.run(None, {sess.get_inputs()[0].name: x})
    stop.set()
    th.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    os_peak = _process_peak_bytes()
    return {"peak_memory_mb": (peak[0] - base_rss) / 2 ** 20,
            "process_peak_mb": None if os_peak is None else os_peak / 2 ** 20}


def measure_peak_memory(onnx_bytes, resolution, threads=1):
    """Peak inference memory (MB) measured in an isolated process; None if unsupported."""
    fd, path = tempfile.mkstemp(suffix=".onnx")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(onnx_bytes)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-m", "q2_nas.hardware", "--onnx", path,
                              "--res", str(resolution), "--threads", str(threads), "--child"],
                             capture_output=True, text=True, cwd=root, timeout=120)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else {"peak_memory_mb": None, "process_peak_mb": None}
    except Exception:
        return {"peak_memory_mb": None, "process_peak_mb": None}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def profile_model(model, resolution, threads=1, with_memory=True, ort_runs=100):
    """Full hardware profile of a candidate network (all measured)."""
    onnx_bytes = export_onnx_bytes(model, resolution)
    prof = {
        "params": sum(p.numel() for p in model.parameters()),
        "onnx_size_mb": len(onnx_bytes) / 2 ** 20,
        "state_dict_size_mb": state_dict_size_bytes(model) / 2 ** 20,
    }
    ort_lat = measure_latency_ort(onnx_bytes, resolution, threads, runs=ort_runs)
    prof.update({"latency_ms": ort_lat["median"], "latency_p95_ms": ort_lat["p95"]})
    prof["torch_latency_ms"] = measure_latency_torch(model, resolution, threads)["median"]
    if with_memory:
        prof.update(measure_peak_memory(onnx_bytes, resolution, threads))
    return prof, onnx_bytes


def check_constraints(prof, constraints):
    """Return (ok, list_of_violations). Missing measurements (None) never reject."""
    violations = []
    if prof["latency_ms"] > constraints["max_latency_ms"]:
        violations.append(f"latency {prof['latency_ms']:.2f}>{constraints['max_latency_ms']}ms")
    if prof["onnx_size_mb"] > constraints["max_size_mb"]:
        violations.append(f"size {prof['onnx_size_mb']:.2f}>{constraints['max_size_mb']}MB")
    mem = prof.get("peak_memory_mb")
    if mem is not None and mem > constraints["max_memory_mb"]:
        violations.append(f"memory {mem:.1f}>{constraints['max_memory_mb']}MB")
    return len(violations) == 0, violations


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Measure ONNX model latency / peak memory on CPU")
    p.add_argument("--onnx", required=True)
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        print(json.dumps(_peak_memory_child(a.onnx, a.res, a.threads)))
    else:
        lat = measure_latency_ort(a.onnx, a.res, a.threads)
        with open(a.onnx, "rb") as f:
            mem = measure_peak_memory(f.read(), a.res, a.threads)
        print(json.dumps({"latency": lat, "size_mb": os.path.getsize(a.onnx) / 2 ** 20, **mem},
                         indent=2))
