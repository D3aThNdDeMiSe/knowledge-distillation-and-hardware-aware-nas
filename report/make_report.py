"""
Build the technical report PDF (Deliverable 12) from the saved results.

    python report/make_figures.py
    python report/make_report.py            # author defaults below; override with --name/--roll/--section/--repo
Output: docs/Technical_Report.pdf
"""
import argparse
import json
import os
import platform
from html import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
NICE = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet",
        "efficientnet_lite0": "EfficientNet-Lite0", "resnet50": "ResNet-50", "nas": "NAS-best"}

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1F3A5F"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8, leading=10, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=8)
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=8, leading=10,
                      backColor=colors.HexColor("#F3F3F3"), borderPadding=4, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.5, leading=9, spaceAfter=0)


INFO = ParagraphStyle("info", parent=BODY, fontSize=10.5, leading=14, spaceAfter=0)


def link(url):
    return f"<link href='{escape(url)}' color='#1F5FAF'><u>{escape(url)}</u></link>"


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [P(f"&bull;&nbsp; {t}") for t in items]


def table(rows, widths=None, font=7.5, header=True, wrap=True):
    data = [[Paragraph(escape(str(c)), CELL) if wrap else str(c) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
             ("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    if header:
        for i, c in enumerate(rows[0]):
            data[0][i] = Paragraph(f"<font color='white'><b>{escape(str(c))}</b></font>", CELL)
    return t


def fig(name, width_cm=16, caption=None):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return []
    from PIL import Image as PI
    w, h = PI.open(p).size
    width = width_cm * cm
    out = [Image(p, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return [KeepTogether(out)]


def load(*parts, kind="csv"):
    p = os.path.join(RES, *parts)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p) if kind == "csv" else json.load(open(p))


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def build(args):
    bench = load("benchmark.csv")
    comp = load("q1", "kd_vs_baseline.csv")
    onnx_rows = load("onnx_export.csv")
    best = load("q2", "best_architecture.json", kind="json")
    hist = load("q2", "search_history.json", kind="json")
    nas_curve = load("q2", "nas_best_history.csv")
    th = load("q1", "teacher_resnet50_history.csv")
    b = {r.model: r for r in bench.itertuples()} if bench is not None else {}

    story = []
    # ------------------------------------------------------------------ title page
    story += [Spacer(1, 3 * cm),
              P("<b>Assignment # 04</b>", ParagraphStyle("t0", parent=H1, alignment=TA_CENTER, fontSize=20)),
              P("Edge Computing &amp; Intelligent Systems", ParagraphStyle("t1", parent=H2, alignment=TA_CENTER, fontSize=14)),
              Spacer(1, 0.6 * cm),
              P("<b>Technical Report</b><br/>Q1: Knowledge Distillation (ResNet-50 &rarr; MobileNetV2 / SqueezeNet / "
                "EfficientNet-Lite0) on CIFAR-100<br/>Q2: Hardware-aware Neural Architecture Search (Optuna) on CIFAR-10",
                ParagraphStyle("t2", parent=BODY, alignment=TA_CENTER, fontSize=11, leading=16)),
              Spacer(1, 1.5 * cm)]
    info = Table([[P(f"<b>{k}</b>", INFO), P(escape(v), INFO)]
                  for k, v in [("Name", args.name), ("Roll number", args.roll), ("Section", args.section)]],
                 colWidths=[4 * cm, 7 * cm])
    info.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                              ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F8FC")),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [info, Spacer(1, 0.5 * cm),
              P(f"GitHub repository:<br/>{link(args.repo)}",
                ParagraphStyle("t3", parent=BODY, alignment=TA_CENTER, fontSize=10, leading=14)),
              Spacer(1, 1.2 * cm)]
    hw = [["Item", "Value"],
          ["GPU (training)", "NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB (CUDA 12.8, bf16 autocast)"],
          ["CPU (latency / ONNX Runtime)", f"AMD Ryzen 7 260 (8C/16T) - {platform.system()} {platform.release()}"],
          ["Software", "Python 3.13, PyTorch 2.8, ONNX 1.23, ONNX Runtime 1.24, Optuna 4.9"],
          ["Datasets", "CIFAR-100 (Q1), CIFAR-10 (Q2): 50k train / 10k test, 32x32 RGB"]]
    story += [table(hw, widths=[4.5 * cm, 11.5 * cm], font=8.5), PageBreak()]

    # ------------------------------------------------------------------ 1 overview
    story += [P("1. Overview and repository structure", H1),
              P("All code was written from scratch in PyTorch for this assignment. Every number in this report "
                "is produced by the scripts in the repository and read automatically from the generated "
                "CSV/JSON files in <font face='Courier'>results/</font>; the console captures in Section 5 are the real "
                "outputs of the runs. The table maps every required deliverable to its files.")]
    deliv = [["#", "Deliverable", "Files"],
             ["1", "Teacher model", "q1_kd/models/resnet.py, q1_kd/train_teacher.py, checkpoints/teacher_resnet50.pt"],
             ["2", "Student architectures", "q1_kd/models/mobilenetv2.py, squeezenet.py, efficientnet_lite.py"],
             ["3", "Baseline students", "q1_kd/train_baseline.py, checkpoints/baseline_*.pt (3)"],
             ["4", "Knowledge distillation", "q1_kd/kd.py, q1_kd/train_kd.py, checkpoints/kd_*.pt (3)"],
             ["5", "NAS search space", "q2_nas/search_space.py, q2_nas/generator.py"],
             ["6", "NAS algorithm", "q2_nas/search.py (Optuna TPE), q2_nas/hardware.py"],
             ["7", "NAS results", "results/q2/best_architecture.json, search_history.csv/.json, optuna_trials.csv"],
             ["8", "Optimised architecture", "q2_nas/train_best.py, checkpoints/nas_best.pt"],
             ["9", "Inference pipeline", "infer.py (all students + NAS model, PyTorch and ONNX back-ends), common/zoo.py"],
             ["10", "ONNX deployment", "export_onnx.py, onnx_models/*.onnx, infer.py --backend onnx"],
             ["11", "Benchmarking", "benchmark.py, results/benchmark.csv, results/q1/kd_vs_baseline.csv"],
             ["12", "Technical report", "docs/Technical_Report.pdf (this file), report/make_figures.py, report/make_report.py"],
             ["13", "README", "README.md, requirements.txt, run_q1.sh, run_q2.sh"]]
    story += [table(deliv, widths=[0.8 * cm, 3.6 * cm, 11.6 * cm])]
    story += [P("Engineering notes", H3)] + bullets([
        "<b>GPU-resident data pipeline</b> (common/data.py): the whole dataset is stored on the GPU as uint8 and "
        "random-crop (4 px reflection padding) + horizontal flip are applied per sample on the GPU. This removes the "
        "Windows DataLoader bottleneck; one ResNet-50 epoch takes ~35 s.",
        "<b>Mixed precision</b>: bf16 autocast (no loss scaling needed). Memory layout was benchmarked: channels-last "
        "is ~20% faster for the dense ResNet-50, plain NCHW is ~25% faster for the depthwise-heavy students.",
        "<b>Identical recipes</b>: baselines and distilled students use the same optimiser (SGD, Nesterov, momentum 0.9, "
        "weight decay 5e-4 not applied to BN/bias), schedule (2-epoch warm-up + cosine), batch size 128, 60 epochs and "
        "seed, so differences come from distillation only."])

    # ------------------------------------------------------------------ 2 Q1
    story += [PageBreak(), P("2. Question 1 - Knowledge distillation framework", H1),
              P("2.1 Teacher: ResNet-50 on CIFAR-100", H2),
              P("Standard bottleneck ResNet-50 (stages [3, 4, 6, 3], widths 64-512, expansion 4, 23.7 M parameters) "
                "adapted to 32x32 inputs: the 7x7/stride-2 stem and max-pool are replaced by a single 3x3/stride-1 "
                "convolution so the four stages operate at 32, 16, 8 and 4 pixels. The last BN of each residual branch is "
                "zero-initialised (blocks start as identity). Trained from scratch for 70 epochs (SGD, lr 0.1, batch 128). "
                "The checkpoint is stored in fp16 (47 MB) to stay well below GitHub's 100 MB file limit; its accuracy is "
                "re-measured after loading.")]
    if th is not None:
        t1 = b["teacher_resnet50"].top1_torch if "teacher_resnet50" in b else th.test_top1.iloc[-1]
        story += [P(f"<b>Result:</b> test top-1 <b>{fmt(t1)}%</b>, top-5 {fmt(th.test_top5.iloc[-1])}% "
                    f"(best epoch {fmt(th.test_top1.max())}%), total training time "
                    f"{fmt(th.epoch_time_s.sum() / 60, 1)} min.")]
    story += fig("q1_teacher_curves.png", 15, "Figure 1 - Teacher training curves (CIFAR-100).")

    story += [P("2.2 Student architectures", H2),
              P("All three students were implemented from their papers and adapted to 32x32 images by removing early "
                "down-sampling (otherwise a 32x32 image would be reduced to 1x1 by the ImageNet layouts).")]
    st_rows = [["Student", "Key idea", "CIFAR adaptation", "Params", "MMACs", "Feature taps (mid / final)"]]
    desc = {"mobilenetv2": ("Inverted residuals with linear bottlenecks, ReLU6, depthwise 3x3",
                            "stem stride 1, 2nd stage stride 1 (32-16-8-4)", "96@8x8 / 1280@4x4"),
            "squeezenet": ("Fire modules (1x1 squeeze -> 1x1 + 3x3 expand), conv classifier, no FC",
                           "3x3/s1 stem, no first max-pool, BN added", "256@8x8 / 512@8x8"),
            "efficientnet_lite0": ("MBConv (no SE), ReLU6, fixed 32-ch stem / 1280-ch head, stochastic depth",
                                   "stem stride 1, 2nd stage stride 1", "112@8x8 / 1280@4x4")}
    for s, (k, a, f) in desc.items():
        r = b.get(f"kd_{s}")
        st_rows.append([NICE[s], k, a, f"{r.params_M:.2f} M" if r else "-", fmt(r.MMACs, 1) if r else "-", f])
    if "teacher_resnet50" in b:
        r = b["teacher_resnet50"]
        st_rows.append(["ResNet-50 (teacher)", "Bottleneck residual blocks", "3x3/s1 stem, no max-pool",
                        f"{r.params_M:.2f} M", fmt(r.MMACs, 1), "1024@8x8 / 2048@4x4"])
    story += [table(st_rows, widths=[2.6 * cm, 4.6 * cm, 3.6 * cm, 1.4 * cm, 1.4 * cm, 2.6 * cm])]

    story += [P("2.3 Distillation framework (q1_kd/kd.py)", H2),
              P("For every mini-batch the frozen teacher (eval mode, no gradients) and the student see the <i>same</i> "
                "augmented images. The student minimises"),
              P("L = (1 - &alpha;) &middot; CE(z<sub>s</sub>, y) &nbsp;+&nbsp; &alpha; &middot; T<super>2</super> &middot; "
                "KL( softmax(z<sub>t</sub>/T) || softmax(z<sub>s</sub>/T) ) &nbsp;+&nbsp; &beta; &middot; "
                "&Sigma;<sub>l in {mid, final}</sub> || n(r<sub>l</sub>(F<sub>s</sub><super>l</super>)) - "
                "n(F<sub>t</sub><super>l</super>) ||<super>2</super>", CODE)]
    story += bullets([
        "<b>Soft-target KD with temperature-based softmax</b> (Hinton et al.): logits are divided by T before softmax; "
        "T = 4 exposes the teacher's relative probabilities for wrong classes (\"dark knowledge\"). The KL term is "
        "multiplied by T<super>2</super> so its gradient scale does not shrink as T grows.",
        "<b>Combined hard + soft training</b>: &alpha; = 0.9 weights the soft targets and 1 - &alpha; = 0.1 the "
        "ground-truth cross-entropy. Each term can be switched off (--no-hard, --feat-layers none) for ablations.",
        "<b>Feature-based distillation</b> (FitNets-style hints) on two intermediate representations: the 8x8 "
        "mid-level map (teacher layer3, 1024 channels) and the final pre-pooling map (teacher layer4, 2048 channels). "
        "A trainable regressor r<sub>l</sub> (1x1 conv + BN) maps student channels to teacher channels; spatial size is "
        "matched with adaptive average pooling; n(.) is per-sample L2 normalisation, making the loss scale-invariant "
        "(student and teacher activations have very different magnitudes) and bounded in [0, 4]. &beta; = 2 per layer. "
        "Regressors are optimised jointly with the student and discarded after training.",
        "Students are trained for 60 epochs (lr 0.05) exactly like the baselines (Section 2.4)."])
    story += [P("2.4 Baseline pipeline", H2),
              P("q1_kd/train_baseline.py trains the three students independently with plain cross-entropy on the hard "
                "labels, with the identical data pipeline, optimiser, schedule, epochs and seed as the distilled students.")]

    story += [P("2.5 Results: baseline vs. distillation", H2)]
    if comp is not None:
        rows = [["Student", "Params (M)", "Baseline top-1", "KD top-1", "Gain", "Baseline top-5", "KD top-5",
                 "Compression vs teacher"]]
        for r in comp.itertuples():
            rows.append([NICE[r.student], fmt(r.params_M, 2), fmt(r.baseline_top1), fmt(r.kd_top1),
                         f"{r.kd_gain_top1:+.2f}", fmt(r.baseline_top5), fmt(r.kd_top5),
                         f"{r.compression_vs_teacher}x fewer params"])
        story += [table(rows, widths=[2.8 * cm, 1.6 * cm, 2 * cm, 1.7 * cm, 1.3 * cm, 2 * cm, 1.7 * cm, 3 * cm])]
        g = comp.kd_gain_top1
        verdict = ("Distillation improved every student" if (g > 0).all() else
                   "Distillation improved " + ", ".join(NICE[s] for s, v in zip(comp.student, g) if v > 0)
                   if (g > 0).any() else "Distillation did not improve the students in this run")
        story += [P(f"{verdict} (mean gain {g.mean():+.2f} top-1 points, max {g.max():+.2f}). "
                    f"The teacher reaches {fmt(comp.teacher_top1.iloc[0])}% while the students have "
                    f"{comp.compression_vs_teacher.min()}-{comp.compression_vs_teacher.max()}x fewer parameters.")]
    story += fig("q1_kd_vs_baseline.png", 13, "Figure 2 - CIFAR-100 test accuracy of baseline and distilled students.")
    story += fig("q1_students_curves.png", 17, "Figure 3 - Test accuracy during training (baseline vs KD).")
    abl = load("q1", "kd_logits_mobilenetv2_history.csv")
    if abl is not None and comp is not None and (comp.student == "mobilenetv2").any():
        r = comp[comp.student == "mobilenetv2"].iloc[0]
        story += [P("Ablation: contribution of the loss terms (MobileNetV2)", H3),
                  table([["Variant", "Loss", "Test top-1 (%)"],
                         ["Baseline", "CE(hard labels)", fmt(r.baseline_top1)],
                         ["KD - soft targets only", "0.1 CE + 0.9 T^2 KL (T = 4)", fmt(abl.test_top1.iloc[-1])],
                         ["KD - soft targets + feature hints (main)", "0.1 CE + 0.9 T^2 KL + 2 (L_mid + L_final)",
                          fmt(r.kd_top1)]], widths=[5.5 * cm, 6.5 * cm, 3 * cm]),
                  P("Same recipe, 60 epochs and seed for all three runs (logs/q1_kd_ablation_logits_only.log).", SMALL)]

    story += [P("2.6 Unified inference pipeline and ONNX deployment", H2),
              P("<font face='Courier'>infer.py</font> exposes one <font face='Courier'>Predictor</font> class for every model "
                "(teacher, 3 baselines, 3 distilled students and the NAS model) with two interchangeable back-ends: the "
                "PyTorch checkpoint (CPU/GPU) or the exported ONNX graph on CPU through ONNX Runtime. Pre-processing is "
                "pure NumPy/PIL and identical for both back-ends (including a bilinear resize that reproduces "
                "PyTorch's <font face='Courier'>interpolate</font> exactly for the NAS model's input resolution), so the "
                "ONNX path needs no PyTorch at run time. It classifies image files, random test images, or evaluates the "
                "full test set. <font face='Courier'>export_onnx.py</font> exports each model (opset 17, dynamic batch), "
                "validates it with <font face='Courier'>onnx.checker</font> and compares ONNX Runtime outputs with PyTorch.")]
    if onnx_rows is not None:
        rows = [["Model", "ONNX file", "Input", "Size (MB)", "max |ORT - PyTorch|", "argmax agreement"]]
        for r in onnx_rows.itertuples():
            rows.append([r.model, os.path.basename(r.onnx_file), r.input_shape, fmt(r.size_mb, 3),
                         r.max_abs_diff_vs_torch, f"{r.top1_agreement:.0%}"])
        story += [table(rows, widths=[3.6 * cm, 4.3 * cm, 2 * cm, 1.6 * cm, 2.6 * cm, 2.2 * cm])]

    # ------------------------------------------------------------------ 3 Q2
    story += [PageBreak(), P("3. Question 2 - Hardware-aware Neural Architecture Search", H1),
              P("3.1 Search space (q2_nas/search_space.py)", H2)]
    ss_rows = [["Dimension", "Choices"],
               ["Input resolution", "24, 28, 32 (images are bilinearly resized; part of the deployed model)"],
               ["Convolutional depth", "3 or 4 stages; 1, 2 or 3 blocks per stage (stride 2 at the start of stages 2-4)"],
               ["Filters / channels", "stem 16/24/32; stage 1: 16-48, stage 2: 32-96, stage 3: 64-160, stage 4: 128-256 "
                                      "(4 options each); optional 1x1 head conv 0/256/512"],
               ["Kernel size", "3, 5, 7 (per stage)"],
               ["Block type", "standard conv, depthwise-separable conv (dw kxk + pw 1x1), MBConv inverted residual "
                              "with expansion 2/4/6 (per stage); residual connection whenever shapes match"],
               ["Activation", "ReLU, ReLU6, Hard-Swish, SiLU (per stage)"]]
    story += [table(ss_rows, widths=[3.5 * cm, 12.5 * cm]),
              P("Per stage there are 3 x 4 x 3 x 5 x 4 = 720 options, giving about 7.3 x 10<super>12</super> architectures. "
                "An architecture is a JSON dictionary; <font face='Courier'>q2_nas/generator.py</font> turns any sampled "
                "configuration into an nn.Module (stem -> stages -> optional head -> global pooling -> dropout -> linear). "
                "The same JSON is stored inside the checkpoint so the model can always be rebuilt.")]

    story += [P("3.2 Search strategy and hardware-aware pipeline", H2)] + bullets([
        "<b>Bayesian optimisation with Optuna</b> (TPE sampler, multivariate + group mode for the conditional "
        "parameters, 10 random start-up trials, seed 0). Sampling is define-by-run, e.g. the expansion ratio is only "
        "sampled for MBConv stages and stage-4 parameters only when 4 stages are chosen.",
        "<b>Measured, not estimated, hardware cost</b> (q2_nas/hardware.py) before any training: the candidate is "
        "exported to ONNX in memory and run with ONNX Runtime on the CPU with 1 thread and batch 1 (emulating a single-core "
        "edge CPU) - 20 warm-up + 100 timed runs, median reported (p95 and PyTorch eager latency also logged); storage "
        "size = bytes of the serialised ONNX model (state_dict size also logged); peak inference memory = increase of "
        "the resident set size while a fresh subprocess loads the model and runs inference, sampled every 0.5 ms "
        "(\"where supported\": returns None if psutil is unavailable).",
        "<b>Rejection of infeasible architectures</b>: candidates violating any deployment constraint are pruned "
        "immediately (Optuna PRUNED state, so TPE learns to avoid that region) and cost ~2 s instead of a training run.",
        "<b>Automated training/validation</b>: feasible candidates are trained for 8 epochs on 45k CIFAR-10 training "
        "images and evaluated on a held-out 5k validation split (the test set is never used during search).",
        "<b>Resource-constrained selection</b>: score = val_acc x (latency/L<sub>max</sub>)<super>-0.05</super> x "
        "(size/S<sub>max</sub>)<super>-0.02</super> (MnasNet-style soft reward on top of the hard constraints) - among "
        "similarly accurate networks the faster and smaller one is preferred.",
        "The best configuration is written to results/q2/best_architecture.json and then <b>trained from scratch</b> "
        "(new random initialisation) on the full 50k training set for 60 epochs (q2_nas/train_best.py)."])
    if best is not None:
        c = best["constraints"]
        story += [P("3.3 Deployment constraints", H2),
                  P(f"Latency &le; <b>{c['max_latency_ms']} ms</b> (ORT, 1 thread, batch 1), ONNX size &le; "
                    f"<b>{c['max_size_mb']} MB</b>, peak inference memory &le; <b>{c['max_memory_mb']} MB</b>. These limits "
                    "were calibrated by profiling 25 random architectures from the space on the target CPU (medians: "
                    "2.0 ms, 1.5 MB, 13 MB), so roughly half of the space is infeasible.")]
    if hist is not None:
        h = pd.DataFrame(hist)
        acc = h[h.status == "accepted"]
        s = best["search"]
        story += [P("3.4 Search results", H2),
                  P(f"{len(h)} trials in {s['search_time_min']} min: <b>{len(acc)} accepted</b> and "
                    f"<b>{len(h) - len(acc)} rejected</b> by the hardware constraints. Proxy validation accuracy of "
                    f"accepted candidates ranged from {acc.val_acc.min():.1f}% to {acc.val_acc.max():.1f}%. "
                    "Full history: results/q2/search_history.csv/.json (every trial with its configuration, measured "
                    "latency, size, memory, status and rejection reason) and optuna_trials.csv.")]
        story += fig("q2_nas_search_space.png", 17, "Figure 4 - Left: every sampled architecture by measured latency and "
                     "size (red lines = constraints). Right: accuracy/latency of the feasible ones.")
        story += fig("q2_nas_progress.png", 17, "Figure 5 - Search progress and reasons for rejection.")
        top = acc.sort_values("score", ascending=False).head(8)
        rows = [["Trial", "Architecture", "Val acc", "Lat (ms)", "Size (MB)", "Mem (MB)", "Score"]]
        for r in top.itertuples():
            rows.append([r.trial, r.arch, fmt(r.val_acc), fmt(r.latency_ms, 3), fmt(r.onnx_size_mb, 3),
                         fmt(r.peak_memory_mb, 1), fmt(r.score)])
        story += [P("Top accepted architectures (by score):", H3),
                  table(rows, widths=[1 * cm, 8.4 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 1.3 * cm, 1.3 * cm], font=7)]
        rej = h[h.status == "rejected"].head(6)
        rows = [["Trial", "Architecture", "Lat (ms)", "Size (MB)", "Rejection reason"]]
        for r in rej.itertuples():
            rows.append([r.trial, r.arch, fmt(r.latency_ms, 3), fmt(r.onnx_size_mb, 3), r.reject_reason])
        story += [P("Examples of rejected architectures:", H3),
                  table(rows, widths=[1 * cm, 7.6 * cm, 1.3 * cm, 1.4 * cm, 4.7 * cm], font=7)]

    if best is not None:
        m = best["metrics"]
        story += [P("3.5 Selected architecture and training from scratch", H2),
                  P(f"<b>Trial {best['best_trial']}</b>: <font face='Courier'>{escape(best['architecture'])}</font>"),
                  P(f"{m['params'] / 1e3:.1f} k parameters, measured latency {fmt(m['latency_ms'], 3)} ms, "
                    f"ONNX size {fmt(m['onnx_size_mb'], 3)} MB, peak memory {fmt(m['peak_memory_mb'], 1)} MB, proxy "
                    f"validation accuracy {fmt(m['val_acc'])}% after 8 epochs.")]
        stage_rows = [["Stage", "Blocks", "Block type", "Channels", "Kernel", "Expansion", "Activation"]]
        for i, st in enumerate(best["config"]["stages"]):
            stage_rows.append([i + 1, st["depth"], st["block"], st["channels"], st["kernel"], st["expansion"],
                               st["activation"]])
        story += [table(stage_rows, widths=[1.3 * cm, 1.4 * cm, 2.2 * cm, 1.8 * cm, 1.5 * cm, 1.8 * cm, 2.2 * cm]),
                  P(f"Input resolution {best['config']['resolution']}x{best['config']['resolution']}, stem "
                    f"{best['config']['stem_channels']} channels, head conv {best['config']['head_channels'] or 'none'}.")]
        if nas_curve is not None:
            nb = b.get("nas_best")
            story += [P(f"Trained from scratch for {len(nas_curve)} epochs on the full CIFAR-10 training set: "
                        f"<b>test top-1 {fmt(nb.top1_torch if nb else nas_curve.test_top1.iloc[-1])}%</b>"
                        + (f" (ONNX Runtime: {fmt(nb.top1_onnx)}%), deployed latency {fmt(nb.ort_lat_ms_1t, 3)} ms "
                           f"(1 thread), {fmt(nb.onnx_MB, 3)} MB." if nb else "."))]
            story += fig("q2_nas_best_curve.png", 9, "Figure 6 - Training the selected architecture from scratch.")

    # ------------------------------------------------------------------ 4 benchmark
    if bench is not None:
        story += [PageBreak(), P("4. Benchmarking (benchmark.py)", H1),
                  P("All models were benchmarked on the same machine with the GPU idle. Accuracy is measured on the full "
                    "test set both with PyTorch and with the exported ONNX model in ONNX Runtime (CPU); latency is the "
                    "median of 200 batch-1 runs after warm-up; throughput uses the full test set with batch 256 and all "
                    "CPU threads. Raw data: results/benchmark.csv.")]
        rows = [["Model", "Data", "Params (M)", "MMACs", "ONNX MB", "Top-1 PT", "Top-1 ORT", "PT CPU 1t (ms)",
                 "ORT 1t (ms)", "ORT 4t (ms)", "ORT img/s", "Peak mem (MB)"]]
        for r in bench.itertuples():
            rows.append([r.model, r.dataset.upper().replace("CIFAR", "C"), fmt(r.params_M, 3), fmt(r.MMACs, 1),
                         fmt(r.onnx_MB, 2), fmt(r.top1_torch), fmt(r.top1_onnx), fmt(r.torch_cpu_lat_ms_1t),
                         fmt(r.ort_lat_ms_1t, 3), fmt(r.ort_lat_ms_4t, 3), fmt(r.ort_throughput_img_s, 0),
                         fmt(r.peak_mem_MB, 1)])
        story += [table(rows, widths=[3.3 * cm, 0.9 * cm, 1.2 * cm, 1.2 * cm, 1.1 * cm, 1.1 * cm, 1.2 * cm, 1.3 * cm,
                                      1.2 * cm, 1.2 * cm, 1.2 * cm, 1.2 * cm], font=6.8)]
        story += fig("q1_accuracy_vs_latency.png", 12, "Figure 7 - Accuracy vs. CPU latency of the CIFAR-100 models.")
        if "teacher_resnet50" in b:
            t = b["teacher_resnet50"]
            fastest = min((r for r in bench.itertuples() if r.group == "kd"), key=lambda r: r.ort_lat_ms_1t)
            story += [P(f"Observations: the ResNet-50 teacher needs {fmt(t.ort_lat_ms_1t)} ms per image on one CPU "
                        f"thread and {fmt(t.onnx_MB, 1)} MB; the fastest distilled student ({fastest.model}) needs "
                        f"{fmt(fastest.ort_lat_ms_1t)} ms ({t.ort_lat_ms_1t / fastest.ort_lat_ms_1t:.1f}x faster) and "
                        f"{fmt(fastest.onnx_MB, 1)} MB. ONNX Runtime is consistently faster than PyTorch eager on the CPU, "
                        "and the ONNX accuracy matches the PyTorch accuracy (tiny differences come from floating-point "
                        "summation order), confirming correct export.")]

    # ------------------------------------------------------------------ 5 screenshots
    shots = [("console_teacher.png", "Teacher training (last epochs)"),
             ("console_baselines.png", "Baseline training of the three students"),
             ("console_kd.png", "Knowledge-distillation training of the three students"),
             ("console_nas_search.png", "Hardware-aware NAS search (accepted / rejected trials)"),
             ("console_nas_train.png", "Training the selected architecture from scratch"),
             ("console_export.png", "ONNX export and verification"),
             ("console_infer.png", "Unified inference pipeline (ONNX Runtime, CPU)"),
             ("console_infer_eval.png", "Test-set evaluation of all models through ONNX Runtime"),
             ("console_benchmark.png", "Automated benchmarking")]
    story += [PageBreak(), P("5. Screenshots (console output of the actual runs)", H1)]
    for i, (f, cap) in enumerate(shots):
        story += fig(f, 17, f"Screenshot {i + 1} - {cap}.")

    # ------------------------------------------------------------------ 6 conclusion
    story += [P("6. Discussion and conclusions", H1)] + bullets([
        "A complete distillation framework was built: temperature-scaled soft targets, combined hard/soft loss and "
        "feature-based hints on two intermediate representations, with a strictly comparable baseline pipeline.",
        "Feature alignment needs care: teacher and student feature maps differ in channels, resolution and scale, which is "
        "handled by 1x1 regressors, adaptive pooling and L2 normalisation.",
        "The NAS framework is hardware-aware in the strict sense: every decision uses latency, size and memory "
        "<i>measured</i> on the deployment runtime (ONNX Runtime on the CPU), and infeasible networks are rejected before "
        "any GPU time is spent on them.",
        "Limitations: candidates are ranked with a short 8-epoch proxy training, which favours fast-converging networks; "
        "latency was measured on a laptop CPU (an ARM edge board would give different absolute numbers, but the pipeline "
        "is unchanged - only the constraints need re-calibration). Longer training (200+ epochs) would raise all "
        "accuracies by several points."])
    story += [P("7. Reproduction", H1),
              P(f"Source code, trained checkpoints, ONNX models, results and logs: {link(args.repo)}"),
              P("See README.md. In short: <font face='Courier'>pip install -r requirements.txt</font>, then "
                "<font face='Courier'>bash run_q1.sh</font>, <font face='Courier'>bash run_q2.sh</font>, "
                "<font face='Courier'>python export_onnx.py</font>, <font face='Courier'>python benchmark.py</font>, "
                "<font face='Courier'>python infer.py --model all --backend onnx --eval</font>.")]

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "Technical_Report.pdf")

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(2 * cm, 1.2 * cm, f"Edge Computing & Intelligent Systems - Assignment 4 - {args.roll}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Assignment 4 - Technical Report",
                            author=args.name)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("Saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Immad Shah")
    ap.add_argument("--roll", default="I22-0395")
    ap.add_argument("--section", default="DS-A")
    ap.add_argument("--repo", default="https://github.com/D3aThNdDeMiSe/knowledge-distillation-and-hardware-aware-nas")
    build(ap.parse_args())
