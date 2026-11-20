# Knowledge Distillation vs. Baselines & Hardware-aware NAS

*Edge Computing & Intelligent Systems — Assignment 4*

**Q1 – Knowledge Distillation:** ResNet-50 teacher → MobileNetV2 / SqueezeNet / EfficientNet-Lite0 students on CIFAR-100
(soft targets with temperature, combined hard + soft loss, feature-based distillation), baseline students without KD,
a unified inference pipeline, and ONNX Runtime CPU deployment.

**Q2 – Hardware-aware NAS:** a searchable CNN space (depth, width, kernel size, depthwise-separable / MBConv blocks,
activations, input resolution), an automatic architecture generator, Bayesian optimisation with Optuna (TPE), CPU latency,
model size and peak memory that are *measured*, rejection of architectures that break deployment constraints, training
the best architecture from scratch on CIFAR-10, and ONNX deployment.

**Author:** Immad Shah (I22-0395), Section DS-A

The technical report is in [`docs/Technical_Report.pdf`](docs/Technical_Report.pdf).

## Results at a glance

**Q1 — CIFAR-100, 60 epochs per student, identical recipe for baseline and KD (test top-1, %)**

| Model | Params | Baseline | Distilled (KD) | Gain | ONNX size | CPU latency (ORT, 1 thread) |
|---|---|---|---|---|---|---|
| MobileNetV2 | 2.35 M | 74.56 | **77.76** | +3.20 | 9.0 MB | 3.72 ms |
| SqueezeNet | 0.78 M | 71.61 | **74.07** | +2.46 | 3.0 MB | 2.86 ms |
| EfficientNet-Lite0 | 3.50 M | 75.48 | **78.48** | +3.00 | 13.3 MB | 4.88 ms |
| ResNet-50 teacher | 23.71 M | 78.12 | — | — | 90.3 MB | 40.70 ms |

* Distillation improves every student by 2.46–3.20 points; the distilled EfficientNet-Lite0 (78.48%) even beats its teacher (78.12%) with 6.8x fewer parameters.
* Ablation (MobileNetV2): baseline 74.56% → soft targets only 77.36% → soft targets + feature hints 77.76%.
* Distilled students run 8.3–14.2x faster than the teacher on one CPU thread (ONNX Runtime).

**Q2 — hardware-aware NAS on CIFAR-10**

* Optuna TPE, 50 trials: 35 accepted, 15 rejected for breaking a deployment constraint (latency ≤ 2.0 ms, ONNX size ≤ 1.5 MB, peak memory ≤ 16 MB, all measured on the CPU).
* Selected (trial 49): `r32 stem32 | ds-k5-relu6 x3 c32 | conv-k5-hswish x1 c64 | mb4-k7-hswish x3 c64 | head0`
* Trained from scratch for 60 epochs: **93.06% test top-1** (ONNX Runtime 93.07%), 198 k parameters, 0.76 MB, 1.44 ms per image on one CPU thread, 11.5 MB peak memory.

Full tables: `results/benchmark.csv`, `results/q1/kd_vs_baseline.csv`, `results/q2/`. Figures: `results/figures/`.


---

## 1. Repository structure

```
.
├── common/                 shared code
│   ├── data.py             CIFAR download/cache + GPU-resident loader with on-GPU augmentation
│   ├── utils.py            training loop, LR schedule, evaluation, checkpoint / CSV helpers
│   └── zoo.py              rebuilds ANY trained model (Q1 or Q2) from its checkpoint
├── q1_kd/                  Question 1
│   ├── models/             resnet.py (teacher), mobilenetv2.py, squeezenet.py, efficientnet_lite.py
│   ├── kd.py               distillation framework (soft targets, temperature, hard+soft, feature KD)
│   ├── train_teacher.py    trains ResNet-50 on CIFAR-100
│   ├── train_baseline.py   trains the 3 students WITHOUT distillation
│   └── train_kd.py         trains the 3 students WITH distillation
├── q2_nas/                 Question 2
│   ├── search_space.py     searchable architecture space + samplers
│   ├── generator.py        config dict -> nn.Module
│   ├── hardware.py         measured CPU latency (ONNX Runtime), model size, peak memory, constraints
│   ├── search.py           Optuna TPE hardware-aware search on CIFAR-10
│   └── train_best.py       trains the selected architecture from scratch
├── infer.py                unified inference pipeline (all models; PyTorch or ONNX Runtime back-end)
├── export_onnx.py          exports every model to ONNX and checks it against PyTorch
├── benchmark.py            automated benchmarking -> results/benchmark.csv
├── report/                 make_figures.py + make_report.py -> docs/Technical_Report.pdf
├── docs/                   Technical_Report.pdf
├── run_q1.sh / run_q2.sh   complete pipelines
├── checkpoints/            trained models (*.pt)
├── onnx_models/            exported models (*.onnx) + model_cards.json
├── results/                CSV/JSON results, NAS history, figures
├── logs/                   console logs of the actual runs
├── samples/                a few CIFAR test images for the inference demo
└── assets/class_names.json
```

## 2. Installation

Python 3.10+ is required (tested with Python 3.13, PyTorch 2.8 + CUDA 12.8, Windows 11, RTX 5060 Laptop GPU).

```bash
# 1) PyTorch with CUDA (choose the wheel for your CUDA version; CPU-only also works for inference)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2) everything else
pip install -r requirements.txt
```

The datasets (CIFAR-100 for Q1, CIFAR-10 for Q2) are **downloaded automatically** the first time a script needs them
(Hugging Face mirror of the official datasets, about 280 MB) and cached in `data/` as `.npz` files.

## 3. Step-by-step execution

All commands are run from the repository root. Each step writes its console output to `logs/` when it's run through the
`run_*.sh` scripts.

### Question 1 — Knowledge distillation (CIFAR-100)

```bash
# 1. Teacher: ResNet-50, 70 epochs              -> checkpoints/teacher_resnet50.pt
python -m q1_kd.train_teacher --epochs 70
# 2. Baselines: 3 students, CE only, 60 epochs  -> checkpoints/baseline_{mobilenetv2,squeezenet,efficientnet_lite0}.pt
python -m q1_kd.train_baseline --epochs 60
# 3. Distillation: 3 students, 60 epochs        -> checkpoints/kd_{mobilenetv2,squeezenet,efficientnet_lite0}.pt
python -m q1_kd.train_kd --epochs 60 --T 4 --alpha 0.9 --beta 2 --feat-layers mid final
```
or everything at once: `bash run_q1.sh` (about 4 h on an RTX 5060 Laptop GPU: teacher 48 min, baselines 80 min, KD 100 min).

Loss-term ablation reported in the PDF (soft targets only, no feature hints):
`python -m q1_kd.train_kd --models mobilenetv2 --epochs 60 --feat-layers none --tag kd_logits` -> `checkpoints/kd_logits_mobilenetv2.pt`, `logs/q1_kd_ablation_logits_only.log`.

Useful KD options: `--T` (temperature), `--alpha` (soft-target weight; hard-label weight = 1-alpha),
`--beta` (feature-loss weight), `--feat-layers mid final | none`, `--no-hard`, `--tag` (prefix for ablation runs).

### Question 2 — Hardware-aware NAS (CIFAR-10)

```bash
# 4. Search: 50 Optuna trials, 8 proxy epochs each -> results/q2/{best_architecture.json, search_history.csv/.json}
python -m q2_nas.search --trials 50 --epochs 8 --max-latency-ms 2.0 --max-size-mb 1.5 --max-memory-mb 16
# 5. Train the selected architecture from scratch   -> checkpoints/nas_best.pt
python -m q2_nas.train_best --epochs 60
```
or `bash run_q2.sh` (about 35 min: search 23 min + final training 10 min). Measure any ONNX model on its own:
`python -m q2_nas.hardware --onnx onnx_models/nas_best.onnx --res 32`.

### Deployment, inference and benchmarking

```bash
# 6. Export all trained models to ONNX + numerical check vs PyTorch -> onnx_models/*.onnx
python export_onnx.py
# 7. Unified inference (any model id, or 'all'; --backend onnx | torch)
python infer.py --model kd_mobilenetv2 nas_best --backend onnx --images samples/cifar100 samples/cifar10
python infer.py --model kd_squeezenet --backend torch --random 5
python infer.py --model all --backend onnx --eval            # full test-set accuracy of every model on CPU
# 8. Benchmark everything (accuracy, params, MACs, size, CPU latency, throughput, peak memory)
python benchmark.py                                           # -> results/benchmark.csv, results/q1/kd_vs_baseline.csv
# 9. Figures + report
python report/make_figures.py
python report/make_report.py                                 # -> docs/Technical_Report.pdf
```

Model ids: `teacher_resnet50`, `baseline_mobilenetv2`, `baseline_squeezenet`, `baseline_efficientnet_lite0`,
`kd_mobilenetv2`, `kd_squeezenet`, `kd_efficientnet_lite0`, `nas_best`.

## 4. Deliverables map

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | Teacher model | `q1_kd/models/resnet.py`, `q1_kd/train_teacher.py`, `checkpoints/teacher_resnet50.pt` |
| 2 | Student architectures | `q1_kd/models/{mobilenetv2,squeezenet,efficientnet_lite}.py` |
| 3 | Baseline student models | `q1_kd/train_baseline.py`, `checkpoints/baseline_*.pt` |
| 4 | Knowledge distillation | `q1_kd/kd.py`, `q1_kd/train_kd.py`, `checkpoints/kd_*.pt` (+ ablation `kd_logits_mobilenetv2.pt`) |
| 5 | NAS search space | `q2_nas/search_space.py`, `q2_nas/generator.py` |
| 6 | NAS algorithm | `q2_nas/search.py` (Optuna TPE) + `q2_nas/hardware.py` |
| 7 | NAS results | `results/q2/best_architecture.json`, `results/q2/search_history.{csv,json}`, `results/q2/optuna_trials.csv` |
| 8 | Optimized architecture | `q2_nas/train_best.py`, `checkpoints/nas_best.pt` |
| 9 | Inference pipeline | `infer.py`, `common/zoo.py` |
| 10 | ONNX deployment | `export_onnx.py`, `onnx_models/*.onnx`, `infer.py --backend onnx` |
| 11 | Benchmarking | `benchmark.py`, `results/benchmark.csv`, `results/q1/kd_vs_baseline.csv` |
| 12 | Technical report | `docs/Technical_Report.pdf` |
| 13 | README | this file, `requirements.txt` |

## 5. Implementation notes

* **Data pipeline:** the whole dataset sits on the GPU (uint8). Random crop (4-px reflection padding) and horizontal flip run
  per sample on the GPU, which avoids the slow multi-process DataLoader on Windows.
* **Training recipe (all models):** SGD + Nesterov (momentum 0.9), weight decay 5e-4 (not on BN/bias), linear warm-up +
  cosine decay, batch 128, bf16 autocast. Baselines and KD students use identical settings, so any accuracy difference
  comes from distillation.
* **KD loss:** `(1-α)·CE + α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) + β·Σ_l ‖n(r_l(F_s)) − n(F_t)‖²` with T=4, α=0.9, β=2,
  hints at the 8×8 mid-level map and the final pre-pool map; `r_l` = 1×1 conv + BN regressor, `n` = per-sample L2 norm.
* **Hardware measurements:** ONNX Runtime CPU latency at batch 1 on 1 thread (median of 100 runs after warm-up). Size is
  the serialised ONNX file. Peak memory is the RSS increase of a fresh process while it loads the model and runs it.
  Constraint violations are rejected *before* training (pruned Optuna trials).
* **Checkpoint sizes:** the teacher is stored in fp16 (≈47 MB) so every file stays under GitHub's 100 MB limit (no Git LFS
  needed).
