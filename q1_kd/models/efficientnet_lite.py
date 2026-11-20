"""
EfficientNet-Lite0 student (Tan & Le, 2019; "Lite" variant by Google for edge TPUs/CPUs),
adapted to 32x32 inputs.

EfficientNet-Lite differs from EfficientNet-B0 to be quantisation/edge friendly:
  * no Squeeze-and-Excitation blocks,
  * ReLU6 instead of Swish/SiLU,
  * stem (32 ch) and head (1280 ch) are fixed and not scaled.
Building block = MBConv: 1x1 expand -> kxk depthwise -> 1x1 linear projection, with
residual connection + stochastic depth (drop-connect) when shapes match.

Lite0 = width 1.0, depth 1.0 (B0 baseline layout):
   stage: kernel, expand, out_ch, repeats, stride
     1      3       1      16      1       1
     2      3       6      24      2       2 -> 1 (CIFAR)
     3      5       6      40      2       2
     4      3       6      80      3       2
     5      5       6     112      3       1
     6      5       6     192      4       2
     7      3       6     320      1       1
CIFAR adaptation: stem stride 2 -> 1 and stage-2 stride 2 -> 1, giving
32 -> 16 -> 8 -> 4 spatial resolution.

Feature taps for distillation:
  'mid'   = output of stage 5   [B, 112, 8, 8]
  'final' = output of head conv [B, 1280, 4, 4]
"""
import torch
import torch.nn as nn

from .mobilenetv2 import _init_weights


def conv_bn_relu6(in_ch, out_ch, k=3, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, expand, drop_connect=0.0):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.drop_connect = drop_connect
        layers = []
        if expand != 1:
            layers.append(conv_bn_relu6(in_ch, hidden, 1))
        layers += [conv_bn_relu6(hidden, hidden, k, stride, groups=hidden),   # depthwise kxk
                   conv_bn_relu6(hidden, out_ch, 1, act=False)]               # linear projection
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_connect > 0:
            # Stochastic depth: randomly drop the whole residual branch per sample.
            keep = 1.0 - self.drop_connect
            mask = torch.empty(out.shape[0], 1, 1, 1, device=out.device,
                               dtype=out.dtype).bernoulli_(keep)
            out = out / keep * mask
        return x + out


class EfficientNetLite0(nn.Module):
    CFG = [(3, 1, 16, 1, 1), (3, 6, 24, 2, 1), (5, 6, 40, 2, 2), (3, 6, 80, 3, 2),
           (5, 6, 112, 3, 1), (5, 6, 192, 4, 2), (3, 6, 320, 1, 1)]

    def __init__(self, num_classes=100, dropout=0.2, drop_connect_rate=0.1):
        super().__init__()
        self.stem = conv_bn_relu6(3, 32, 3, stride=1)
        total_blocks = sum(c[3] for c in self.CFG)
        in_ch, idx, stages = 32, 0, []
        for k, e, c, n, s in self.CFG:
            blocks = []
            for i in range(n):
                # drop-connect rate grows linearly with depth (as in EfficientNet)
                dc = drop_connect_rate * idx / total_blocks
                blocks.append(MBConv(in_ch, c, k, s if i == 0 else 1, e, dc))
                in_ch = c
                idx += 1
            stages.append(nn.Sequential(*blocks))
        self.features_low = nn.Sequential(*stages[:5])    # ends at 112 ch, 8x8
        self.features_high = nn.Sequential(*stages[5:])   # ends at 320 ch, 4x4
        self.head = conv_bn_relu6(in_ch, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        self.feature_channels = {"mid": 112, "final": 1280}
        _init_weights(self)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        mid = self.features_low(x)
        final = self.head(self.features_high(mid))
        logits = self.classifier(self.pool(final).flatten(1))
        if return_features:
            return logits, {"mid": mid, "final": final}
        return logits
