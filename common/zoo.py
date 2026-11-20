"""
Model zoo: one place that knows how to rebuild ANY trained model of the project from
its checkpoint (Q1 teacher / baseline students / distilled students, Q2 NAS model).
Used by the unified inference pipeline, ONNX export and benchmarking scripts.

Model ids (= checkpoint file names without .pt):
    teacher_resnet50
    baseline_mobilenetv2   baseline_squeezenet   baseline_efficientnet_lite0
    kd_mobilenetv2         kd_squeezenet         kd_efficientnet_lite0
    nas_best
"""
import os

import torch

from common.utils import CKPT_DIR, ONNX_DIR, load_state

STUDENT_ARCHS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
ALL_MODEL_IDS = (["teacher_resnet50"]
                 + [f"baseline_{a}" for a in STUDENT_ARCHS]
                 + [f"kd_{a}" for a in STUDENT_ARCHS]
                 + ["nas_best"])
STUDENT_MODEL_IDS = [m for m in ALL_MODEL_IDS if m.startswith(("baseline_", "kd_"))] + ["nas_best"]


def checkpoint_path(model_id):
    return model_id if model_id.endswith(".pt") else os.path.join(CKPT_DIR, f"{model_id}.pt")


def onnx_path(model_id):
    return os.path.join(ONNX_DIR, f"{model_id}.onnx")


def available_models(ids=None):
    return [m for m in (ids or ALL_MODEL_IDS) if os.path.exists(checkpoint_path(m))]


def load_meta(model_id):
    return torch.load(checkpoint_path(model_id), map_location="cpu", weights_only=False)


def build_model_from_meta(meta):
    if meta["arch"] == "nas":
        from q2_nas.generator import build_from_config
        return build_from_config(meta["config"], meta["num_classes"])
    from q1_kd.models import build_model
    return build_model(meta["arch"], meta["num_classes"])


def load_model(model_id, device="cpu"):
    """Return (model in eval mode, info dict with dataset / resolution / accuracy)."""
    ckpt = load_meta(model_id)
    model = load_state(build_model_from_meta(ckpt), ckpt).to(device).eval()
    info = {k: v for k, v in ckpt.items() if k != "state_dict"}
    info["resolution"] = ckpt.get("config", {}).get("resolution", 32)
    info["model_id"] = os.path.splitext(os.path.basename(checkpoint_path(model_id)))[0]
    return model, info
