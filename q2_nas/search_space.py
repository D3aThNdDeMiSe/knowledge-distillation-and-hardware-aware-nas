"""
Searchable CNN architecture space (Deliverable 5).

An architecture is a plain JSON-serialisable dict ("config"). The space covers:
  * variable input resolution          resolution      in {24, 28, 32}
  * variable convolutional depth       num_stages      in {3, 4}
                                       stage depth     in {1, 2, 3}   (blocks per stage)
  * variable number of filters         stem_channels   in {16, 24, 32}
                                       stage channels  from a per-stage list (below)
                                       head_channels   in {0 (no head conv), 256, 512}
  * different kernel sizes             stage kernel    in {3, 5, 7}
  * block type incl. depthwise-separable convolution
                                       stage block     in {conv, dsconv, mbconv}
                                         conv   = standard kxk conv + BN + act
                                         dsconv = depthwise kxk + pointwise 1x1 (MobileNetV1)
                                         mbconv = 1x1 expand + depthwise kxk + 1x1 linear
                                                  projection (inverted residual, MobileNetV2)
                                       mbconv expansion in {2, 4, 6}
  * different activation functions     stage activation in {relu, relu6, hswish, silu}

The first stage keeps the stem resolution; every following stage halves it.

Size of the space: per stage 3 depths x 4 widths x 3 kernels x 5 block variants
(conv, dsconv, mbconv-e2/e4/e6) x 4 activations = 720 options, so
3 (res) x 3 (stem) x 3 (head) x (720^3 + 720^4) ~= 7.3e12 configurations -
far too many to enumerate -> an automated search strategy is required.

Two samplers are provided:
  * `sample_config_optuna(trial)`  - define-by-run sampling for Optuna (TPE)
  * `sample_config_random(rng)`    - uniform random sampling (used for calibration)
"""
import random

RESOLUTIONS = [24, 28, 32]
NUM_STAGES = [3, 4]
STEM_CHANNELS = [16, 24, 32]
STAGE_CHANNELS = [[16, 24, 32, 48],      # stage 1 (full resolution)
                  [32, 48, 64, 96],      # stage 2 (1/2)
                  [64, 96, 128, 160],    # stage 3 (1/4)
                  [128, 160, 192, 256]]  # stage 4 (1/8), only if num_stages == 4
DEPTHS = [1, 2, 3]
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dsconv", "mbconv"]
EXPANSIONS = [2, 4, 6]
ACTIVATIONS = ["relu", "relu6", "hswish", "silu"]
HEAD_CHANNELS = [0, 256, 512]


def _stage(depth, channels, kernel, block, expansion, activation):
    return {"depth": depth, "channels": channels, "kernel": kernel, "block": block,
            "expansion": expansion if block == "mbconv" else 1, "activation": activation}


def sample_config_optuna(trial):
    """Define-by-run sampling: Optuna records every choice so TPE can model it."""
    cfg = {
        "resolution": trial.suggest_categorical("resolution", RESOLUTIONS),
        "stem_channels": trial.suggest_categorical("stem_channels", STEM_CHANNELS),
        "num_stages": trial.suggest_categorical("num_stages", NUM_STAGES),
        "head_channels": trial.suggest_categorical("head_channels", HEAD_CHANNELS),
        "stages": [],
    }
    for s in range(cfg["num_stages"]):
        block = trial.suggest_categorical(f"s{s}_block", BLOCKS)
        # Conditional parameter: the expansion ratio only exists for MBConv stages.
        expansion = trial.suggest_categorical(f"s{s}_expansion", EXPANSIONS) if block == "mbconv" else 1
        cfg["stages"].append(_stage(
            trial.suggest_categorical(f"s{s}_depth", DEPTHS),
            trial.suggest_categorical(f"s{s}_channels", STAGE_CHANNELS[s]),
            trial.suggest_categorical(f"s{s}_kernel", KERNELS),
            block, expansion,
            trial.suggest_categorical(f"s{s}_activation", ACTIVATIONS)))
    return cfg


def sample_config_random(rng=random):
    cfg = {"resolution": rng.choice(RESOLUTIONS), "stem_channels": rng.choice(STEM_CHANNELS),
           "num_stages": rng.choice(NUM_STAGES), "head_channels": rng.choice(HEAD_CHANNELS),
           "stages": []}
    for s in range(cfg["num_stages"]):
        block = rng.choice(BLOCKS)
        cfg["stages"].append(_stage(rng.choice(DEPTHS), rng.choice(STAGE_CHANNELS[s]),
                                    rng.choice(KERNELS), block, rng.choice(EXPANSIONS),
                                    rng.choice(ACTIVATIONS)))
    return cfg


def describe(cfg):
    """Compact human-readable architecture string, e.g. 'r32 s24 | mb6-k3-a:relu x2 c32 | ...'."""
    parts = [f"r{cfg['resolution']} stem{cfg['stem_channels']}"]
    for st in cfg["stages"]:
        b = {"conv": "conv", "dsconv": "ds", "mbconv": f"mb{st['expansion']}"}[st["block"]]
        parts.append(f"{b}-k{st['kernel']}-{st['activation']} x{st['depth']} c{st['channels']}")
    parts.append(f"head{cfg['head_channels']}")
    return " | ".join(parts)
