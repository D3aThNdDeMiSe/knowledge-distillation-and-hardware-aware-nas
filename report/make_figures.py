"""
Generate all figures for the technical report from the saved results
(results/*.csv|json and logs/*.log). Nothing is re-trained here.

    python report/make_figures.py
Outputs: results/figures/*.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
NICE = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet",
        "efficientnet_lite0": "EfficientNet-Lite0", "resnet50": "ResNet-50"}
C_BASE, C_KD, C_T, C_NAS, C_REJ = "#4C72B0", "#DD8452", "#55A868", "#8172B3", "#B0B0B0"

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def _csv(*parts):
    p = os.path.join(RES, *parts)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def fig_teacher():
    h = _csv("q1", "teacher_resnet50_history.csv")
    if h is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(h.epoch, h.train_acc, label="train (augmented)", color=C_BASE)
    ax[0].plot(h.epoch, h.test_top1, label="test top-1", color=C_T)
    ax[0].plot(h.epoch, h.test_top5, label="test top-5", color=C_T, ls="--")
    ax[0].set(xlabel="epoch", ylabel="accuracy (%)", title="ResNet-50 teacher on CIFAR-100")
    ax[0].legend()
    ax[1].plot(h.epoch, h.train_loss, label="train loss", color=C_BASE)
    ax[1].plot(h.epoch, h.test_loss, label="test loss", color=C_T)
    ax[1].set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    ax[1].legend()
    save(fig, "q1_teacher_curves.png")


def fig_student_curves():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    any_ = False
    for ax, s in zip(axes, STUDENTS):
        b, k = _csv("q1", f"baseline_{s}_history.csv"), _csv("q1", f"kd_{s}_history.csv")
        if b is not None:
            ax.plot(b.epoch, b.test_top1, color=C_BASE, label=f"baseline (best {b.test_top1.max():.1f})")
            any_ = True
        if k is not None:
            ax.plot(k.epoch, k.test_top1, color=C_KD, label=f"KD (best {k.test_top1.max():.1f})")
            any_ = True
        ax.set(title=NICE[s], xlabel="epoch")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("CIFAR-100 test top-1 (%)")
    if any_:
        save(fig, "q1_students_curves.png")
    else:
        plt.close(fig)


def fig_kd_bars():
    c = _csv("q1", "kd_vs_baseline.csv")
    if c is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = range(len(c))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], c.baseline_top1, w, color=C_BASE, label="baseline (CE only)")
    b2 = ax.bar([i + w / 2 for i in x], c.kd_top1, w, color=C_KD, label="knowledge distillation")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.3, f"{r.get_height():.1f}",
                    ha="center", fontsize=8)
    if c.teacher_top1.notna().any():
        ax.axhline(c.teacher_top1.iloc[0], color=C_T, ls="--",
                   label=f"ResNet-50 teacher ({c.teacher_top1.iloc[0]:.1f})")
    ax.set_xticks(list(x), [f"{NICE[s]}\n({p:.2f}M params)" for s, p in zip(c.student, c.params_M)])
    ax.set_ylabel("CIFAR-100 test top-1 (%)")
    ax.set_ylim(min(c.baseline_top1.min(), c.kd_top1.min()) - 10, max(c.teacher_top1.max(), c.kd_top1.max()) + 6)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.set_title("Baseline vs. distilled students")
    save(fig, "q1_kd_vs_baseline.png")


def fig_acc_latency():
    b = _csv("benchmark.csv")
    if b is None:
        return
    q1 = b[b.dataset == "cifar100"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = {"teacher": C_T, "baseline": C_BASE, "kd": C_KD}
    for _, r in q1.iterrows():
        ax.scatter(r.ort_lat_ms_1t, r.top1_onnx, s=30 + 25 * r.params_M ** 0.5 * 4,
                   color=colors[r.group], alpha=0.85, edgecolor="k", lw=0.5)
        ax.annotate(r.model.replace("baseline_", "base/").replace("kd_", "kd/").replace("teacher_", ""),
                    (r.ort_lat_ms_1t, r.top1_onnx), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel="ONNX Runtime CPU latency, batch 1, 1 thread (ms, log)",
           ylabel="CIFAR-100 top-1 (ONNX, %)", title="Accuracy vs. CPU latency (marker size ~ params)")
    for g, c in colors.items():
        ax.scatter([], [], color=c, label=g)
    ax.legend(fontsize=7)
    save(fig, "q1_accuracy_vs_latency.png")


def fig_nas():
    hist_path = os.path.join(RES, "q2", "search_history.json")
    if not os.path.exists(hist_path):
        return
    h = pd.DataFrame(json.load(open(hist_path)))
    best = json.load(open(os.path.join(RES, "q2", "best_architecture.json")))
    cons = best["constraints"]
    acc, rej = h[h.status == "accepted"], h[h.status == "rejected"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter(rej.latency_ms, rej.onnx_size_mb, marker="x", color=C_REJ, label=f"rejected ({len(rej)})")
    sc = ax[0].scatter(acc.latency_ms, acc.onnx_size_mb, c=acc.val_acc, cmap="viridis", s=28,
                       edgecolor="k", lw=0.4, label=f"accepted ({len(acc)})")
    ax[0].axvline(cons["max_latency_ms"], color="r", ls="--", lw=1)
    ax[0].axhline(cons["max_size_mb"], color="r", ls="--", lw=1, label="constraints")
    bm = best["metrics"]
    ax[0].scatter(bm["latency_ms"], bm["onnx_size_mb"], marker="*", s=220, color="gold",
                  edgecolor="k", zorder=5, label="selected")
    ax[0].set(xscale="log", yscale="log", xlabel="measured CPU latency (ms, ORT 1 thread)",
              ylabel="ONNX model size (MB)", title="Hardware-aware search: feasibility")
    plt.colorbar(sc, ax=ax[0], label="val acc (%)")
    ax[0].legend(fontsize=7, loc="upper left")

    ax[1].scatter(acc.latency_ms, acc.val_acc, c=C_NAS, s=25, edgecolor="k", lw=0.4, label="accepted")
    ax[1].scatter(bm["latency_ms"], bm["val_acc"], marker="*", s=220, color="gold", edgecolor="k",
                  zorder=5, label=f"selected (trial {best['best_trial']})")
    # Pareto front (max accuracy for given latency)
    a = acc.sort_values("latency_ms")
    front, top = [], -1
    for _, r in a.iterrows():
        if r.val_acc > top:
            front.append(r)
            top = r.val_acc
    if front:
        f = pd.DataFrame(front)
        ax[1].step(f.latency_ms, f.val_acc, where="post", color="k", lw=0.8, label="Pareto front")
    ax[1].set(xlabel="measured CPU latency (ms)", ylabel="proxy val accuracy (%)",
              title="Accuracy vs latency of feasible candidates")
    ax[1].legend(fontsize=7)
    save(fig, "q2_nas_search_space.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    sc_ = h.score.astype(float)
    ax[0].scatter(h.trial[h.status == "accepted"], sc_[h.status == "accepted"], color=C_NAS, s=18, label="score")
    ax[0].plot(h.trial, sc_.cummax().ffill(), color="k", lw=1.2, label="best so far")
    ax[0].scatter(h.trial[h.status == "rejected"], [sc_.min(skipna=True) - 2] * len(rej), marker="x",
                  color=C_REJ, label="rejected")
    ax[0].axvline(9.5, color="gray", ls=":", lw=1)
    ax[0].text(10, sc_.min(skipna=True), " TPE starts", fontsize=7, color="gray")
    ax[0].set(xlabel="trial", ylabel="score", title="Optuna TPE search progress")
    ax[0].legend(fontsize=7)
    reasons = {"latency": 0, "size": 0, "memory": 0}
    for r in rej.reject_reason:
        for k in reasons:
            if k in r:
                reasons[k] += 1
    ax[1].bar(list(reasons), list(reasons.values()), color=["#C44E52", "#DD8452", "#937860"])
    ax[1].set(ylabel="# rejected trials", title="Constraint violations (a trial can violate several)")
    save(fig, "q2_nas_progress.png")

    t = _csv("q2", "nas_best_history.csv")
    if t is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(t.epoch, t.train_acc, color=C_BASE, label="train")
        ax.plot(t.epoch, t.test_top1, color=C_NAS, label=f"test (final {t.test_top1.iloc[-1]:.2f})")
        ax.set(xlabel="epoch", ylabel="CIFAR-10 accuracy (%)", title="Best NAS architecture trained from scratch")
        ax.legend()
        save(fig, "q2_nas_best_curve.png")


def render_console(text, name, title):
    """Render captured console output as an image (terminal-style)."""
    lines = text.rstrip("\n").splitlines()
    width = max(len(l) for l in lines) if lines else 80
    fig = plt.figure(figsize=(max(6, width * 0.062), 0.18 * len(lines) + 0.5))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 0.995, title, color="#9cdcfe", family="monospace", fontsize=8, va="top", weight="bold")
    fig.text(0.01, 0.995 - 0.3 / (0.18 * len(lines) + 0.5), "\n".join(lines), color="#d4d4d4",
             family="monospace", fontsize=7.2, va="top", linespacing=1.25)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), facecolor=fig.get_facecolor(), dpi=150,
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("  wrote", name)


def console_captures():
    logs = os.path.join(ROOT, "logs")
    spec = [("q1_teacher.log", 12, "console_teacher.png", "$ python -m q1_kd.train_teacher --epochs 70   (last lines)"),
            ("q1_baselines.log", 14, "console_baselines.png", "$ python -m q1_kd.train_baseline --epochs 60   (excerpt)"),
            ("q1_kd.log", 14, "console_kd.png", "$ python -m q1_kd.train_kd --epochs 60   (excerpt)"),
            ("q2_search.log", 22, "console_nas_search.png", "$ python -m q2_nas.search --trials 50 --epochs 8   (last lines)"),
            ("q2_train_best.log", 8, "console_nas_train.png", "$ python -m q2_nas.train_best --epochs 60   (last lines)"),
            ("export_onnx.log", 14, "console_export.png", "$ python export_onnx.py"),
            ("benchmark.log", 16, "console_benchmark.png", "$ python benchmark.py"),
            ("infer_demo.log", 26, "console_infer.png", "$ python infer.py --model kd_mobilenetv2 nas_best --backend onnx --images samples/cifar100 samples/cifar10"),
            ("infer_eval.log", 12, "console_infer_eval.png", "$ python infer.py --model all --backend onnx --eval")]
    for fn, n, out, title in spec:
        p = os.path.join(logs, fn)
        if not os.path.exists(p):
            continue
        lines = [l.rstrip() for l in open(p, encoding="utf-8", errors="replace") if l.strip()]
        if fn in ("q1_baselines.log", "q1_kd.log"):
            # keep the summary lines of every student + a few epoch lines
            keep = [l for l in lines if l.startswith(("===", "Baseline ", "KD ", "Loaded"))
                    or " ep  60/60" in l or " ep   1/60" in l or " ep  30/60" in l]
            lines = keep
        render_console("\n".join(l[:175] for l in lines[-n:]), out, title)


def main():
    print("Generating figures ->", FIG)
    fig_teacher()
    fig_student_curves()
    fig_kd_bars()
    fig_acc_latency()
    fig_nas()
    console_captures()


if __name__ == "__main__":
    sys.exit(main())
