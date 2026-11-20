"""
Hardware-aware Neural Architecture Search with Bayesian optimisation (Optuna TPE)
(Deliverables 6 & 7).

For every trial:
  1. TPE samples an architecture config from the search space (search_space.py).
  2. The generator builds the network (generator.py).
  3. HARDWARE CHECK (before any training, so infeasible nets cost ~2 s, not minutes):
       - export to ONNX, measure real CPU latency with ONNX Runtime (1 thread, batch 1),
       - measure model storage size (ONNX bytes) and peak inference memory
         (isolated subprocess),
       - architectures violating the deployment constraints are REJECTED
         (trial marked PRUNED -> TPE learns to avoid that region).
  4. Feasible nets are trained for a few epochs on CIFAR-10 (45k train split) and
     evaluated on a held-out 5k validation split (the test set is never used here).
  5. Score (resource-constrained selection, MnasNet-style soft reward on top of the hard
     constraints):
         score = val_acc * (latency / L_max)^w_lat * (size / S_max)^w_size
     with small negative exponents, so among similarly accurate nets the faster /
     smaller one wins.
The best accepted architecture is saved to results/q2/best_architecture.json and the
full history to results/q2/search_history.{csv,json}.

Example:
    python -m q2_nas.search --trials 50 --epochs 8
"""
import argparse
import json
import os
import time

import optuna
import torch

from common.data import get_loaders
from common.utils import RESULTS_DIR, ce_loss_fn, fit, get_device, set_seed, write_csv, write_json
from q2_nas.generator import build_from_config
from q2_nas.hardware import DEFAULT_CONSTRAINTS, check_constraints, profile_model
from q2_nas.search_space import describe, sample_config_optuna

OUT_DIR = os.path.join(RESULTS_DIR, "q2")


def main():
    p = argparse.ArgumentParser(description="Hardware-aware NAS (Optuna TPE) on CIFAR-10")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=8, help="training epochs per candidate")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--val-split", type=int, default=5000)
    p.add_argument("--threads", type=int, default=1, help="CPU threads for latency measurement")
    p.add_argument("--max-latency-ms", type=float, default=DEFAULT_CONSTRAINTS["max_latency_ms"])
    p.add_argument("--max-size-mb", type=float, default=DEFAULT_CONSTRAINTS["max_size_mb"])
    p.add_argument("--max-memory-mb", type=float, default=DEFAULT_CONSTRAINTS["max_memory_mb"])
    p.add_argument("--w-lat", type=float, default=-0.05, help="latency exponent in the score")
    p.add_argument("--w-size", type=float, default=-0.02, help="size exponent in the score")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    constraints = {"max_latency_ms": args.max_latency_ms, "max_size_mb": args.max_size_mb,
                   "max_memory_mb": args.max_memory_mb}
    print(f"Deployment constraints: {constraints} | latency threads={args.threads}")

    # Loaders are built once; the resolution is changed per trial (searchable parameter).
    train_loader, val_loader, _ = get_loaders("cifar10", args.batch_size, device,
                                              val_split=args.val_split, seed=args.seed)
    history = []

    def save_history():
        write_csv(os.path.join(args.out_dir, "search_history.csv"),
                  [{k: v for k, v in r.items() if k != "config"} | {"config": json.dumps(r["config"])}
                   for r in history])
        write_json(os.path.join(args.out_dir, "search_history.json"), history)

    def objective(trial):
        cfg = sample_config_optuna(trial)
        model = build_from_config(cfg, num_classes=10)
        t0 = time.time()
        prof, _ = profile_model(model, cfg["resolution"], threads=args.threads)
        ok, violations = check_constraints(prof, constraints)
        row = {"trial": trial.number, "status": "accepted" if ok else "rejected",
               "reject_reason": "; ".join(violations), "arch": describe(cfg),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in prof.items()},
               "val_acc": None, "score": None, "profile_time_s": round(time.time() - t0, 2),
               "train_time_s": None, "config": cfg}
        for k, v in prof.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("config", cfg)
        if not ok:
            history.append(row)
            save_history()
            print(f"[trial {trial.number:3d}] REJECTED ({row['reject_reason']}) :: {row['arch']}",
                  flush=True)
            raise optuna.TrialPruned(row["reject_reason"])

        # ---- feasible: short proxy training on CIFAR-10 ----
        train_loader.resolution = val_loader.resolution = cfg["resolution"]
        model = model.to(device)
        t1 = time.time()
        hist = fit(model, train_loader, val_loader, device, args.epochs, args.lr, args.wd,
                   ce_loss_fn(), f"trial {trial.number}", warmup_epochs=1,
                   print_every=args.epochs)
        val_acc = hist[-1]["test_top1"]
        score = (val_acc * (prof["latency_ms"] / args.max_latency_ms) ** args.w_lat
                 * (prof["onnx_size_mb"] / args.max_size_mb) ** args.w_size)
        row.update({"val_acc": val_acc, "score": round(score, 4),
                    "train_time_s": round(time.time() - t1, 1)})
        trial.set_user_attr("val_acc", val_acc)
        history.append(row)
        save_history()
        print(f"[trial {trial.number:3d}] ACCEPTED val_acc={val_acc:.2f} score={score:.2f} "
              f"lat={prof['latency_ms']:.3f}ms size={prof['onnx_size_mb']:.3f}MB "
              f"mem={prof.get('peak_memory_mb') or float('nan'):.1f}MB :: {row['arch']}", flush=True)
        del model
        torch.cuda.empty_cache()
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10, multivariate=True,
                                         group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="hw_aware_nas")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_start = time.time()
    study.optimize(objective, n_trials=args.trials)
    search_time = time.time() - t_start

    study.trials_dataframe().to_csv(os.path.join(args.out_dir, "optuna_trials.csv"), index=False)
    accepted = [r for r in history if r["status"] == "accepted"]
    if not accepted:
        raise SystemExit("No architecture satisfied the constraints - relax them and re-run.")
    best = max(accepted, key=lambda r: r["score"])
    summary = {
        "best_trial": best["trial"], "architecture": best["arch"], "config": best["config"],
        "metrics": {k: best[k] for k in ("val_acc", "score", "params", "onnx_size_mb",
                                         "state_dict_size_mb", "latency_ms", "latency_p95_ms",
                                         "torch_latency_ms", "peak_memory_mb")},
        "constraints": constraints, "latency_threads": args.threads,
        "search": {"strategy": "Optuna TPE (Bayesian optimisation)", "trials": args.trials,
                   "accepted": len(accepted), "rejected": len(history) - len(accepted),
                   "epochs_per_candidate": args.epochs, "search_time_min": round(search_time / 60, 1),
                   "score": f"val_acc * (lat/{args.max_latency_ms})^{args.w_lat} * "
                            f"(size/{args.max_size_mb})^{args.w_size}"},
    }
    write_json(os.path.join(args.out_dir, "best_architecture.json"), summary)
    print("\n=== NAS finished ===")
    print(f"trials={len(history)} accepted={len(accepted)} rejected={len(history) - len(accepted)} "
          f"time={search_time / 60:.1f} min")
    print(f"BEST trial {best['trial']}: val_acc={best['val_acc']:.2f} score={best['score']:.2f} "
          f"lat={best['latency_ms']:.3f}ms size={best['onnx_size_mb']:.3f}MB\n  {best['arch']}")
    print(f"Saved -> {os.path.join(args.out_dir, 'best_architecture.json')}")


if __name__ == "__main__":
    main()
