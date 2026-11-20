"""
Automated architecture generator (Deliverable 5): turns a sampled config dict
(see search_space.py) into a trainable / exportable nn.Module.

    config -> stem conv -> stage_1 ... stage_N -> [1x1 head conv] -> GAP -> dropout -> FC

Every stage i>0 starts with a stride-2 block (spatial down-sampling); the remaining
blocks of the stage use stride 1 and get a residual connection when shapes match.
"""
import torch.nn as nn

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "hswish": lambda: nn.Hardswish(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
}


def conv_bn(in_ch, out_ch, k, stride=1, groups=1, act=None):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act is not None:
        layers.append(ACTIVATIONS[act]())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Standard (dense) kxk convolution + BN + activation."""

    def __init__(self, in_ch, out_ch, k, stride, act, **_):
        super().__init__()
        self.body = conv_bn(in_ch, out_ch, k, stride, act=act)
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)


class DSConvBlock(nn.Module):
    """Depthwise-separable convolution: depthwise kxk (per-channel) + pointwise 1x1.
    Costs ~ (1/out_ch + 1/k^2) of a dense kxk conv."""

    def __init__(self, in_ch, out_ch, k, stride, act, **_):
        super().__init__()
        self.body = nn.Sequential(conv_bn(in_ch, in_ch, k, stride, groups=in_ch, act=act),
                                  conv_bn(in_ch, out_ch, 1, act=act))
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)


class MBConvBlock(nn.Module):
    """Inverted residual: 1x1 expand -> depthwise kxk -> 1x1 linear projection."""

    def __init__(self, in_ch, out_ch, k, stride, act, expansion=4):
        super().__init__()
        hidden = in_ch * expansion
        self.body = nn.Sequential(conv_bn(in_ch, hidden, 1, act=act),
                                  conv_bn(hidden, hidden, k, stride, groups=hidden, act=act),
                                  conv_bn(hidden, out_ch, 1, act=None))
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)


BLOCKS = {"conv": ConvBlock, "dsconv": DSConvBlock, "mbconv": MBConvBlock}


class NASNet(nn.Module):
    def __init__(self, config, num_classes=10, dropout=0.1):
        super().__init__()
        self.config = config
        self.resolution = config["resolution"]
        first_act = config["stages"][0]["activation"]
        self.stem = conv_bn(3, config["stem_channels"], 3, 1, act=first_act)
        in_ch = config["stem_channels"]
        stages = []
        for i, st in enumerate(config["stages"]):
            blocks = []
            for j in range(st["depth"]):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(BLOCKS[st["block"]](in_ch, st["channels"], st["kernel"], stride,
                                                  st["activation"], expansion=st["expansion"]))
                in_ch = st["channels"]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        last_act = config["stages"][-1]["activation"]
        if config["head_channels"]:
            self.head = conv_bn(in_ch, config["head_channels"], 1, act=last_act)
            in_ch = config["head_channels"]
        else:
            self.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(self.stages(self.stem(x)))
        return self.classifier(self.pool(x).flatten(1))


def build_from_config(config, num_classes=10):
    """Public entry point of the generator."""
    return NASNet(config, num_classes)
