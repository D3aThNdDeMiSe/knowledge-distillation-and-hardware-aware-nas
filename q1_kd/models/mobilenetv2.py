"""
MobileNetV2 student (Sandler et al., 2018) adapted to 32x32 inputs.

Building block = inverted residual: 1x1 expansion -> 3x3 depthwise -> 1x1 linear
projection (no activation after the projection = "linear bottleneck"), with a
skip connection when stride == 1 and in/out channels match.

CIFAR adaptation: the stem conv and the first stride-2 stage use stride 1, so the
network down-samples 32 -> 16 -> 8 -> 4 (three times) instead of five times.

Feature taps for distillation:
  'mid'   = output of the 96-channel stage   [B, 96, 8, 8]
  'final' = output of the 1x1 head conv      [B, 1280, 4, 4]
"""
import torch.nn as nn


def conv_bn_act(in_ch, out_ch, k=3, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_act(in_ch, hidden, k=1))                      # expand
        layers.append(conv_bn_act(hidden, hidden, k=3, stride=stride, groups=hidden))  # depthwise
        layers.append(conv_bn_act(hidden, out_ch, k=1, act=False))            # linear projection
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class MobileNetV2(nn.Module):
    # (expansion t, output channels c, repeats n, first stride s) -- paper Table 2,
    # with the 2nd stage stride changed 2 -> 1 for 32x32 inputs.
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        in_ch = int(32 * width_mult)
        self.stem = conv_bn_act(3, in_ch, 3, stride=1)
        stages = []
        for t, c, n, s in self.CFG:
            out_ch = int(c * width_mult)
            blocks = [InvertedResidual(in_ch if i == 0 else out_ch, out_ch, s if i == 0 else 1, t)
                      for i in range(n)]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        # stages[0..4] run at 32,32,16,8,8 px -> 'mid' tap after stage index 4 (96 ch @ 8x8)
        self.features_low = nn.Sequential(*stages[:5])
        self.features_high = nn.Sequential(*stages[5:])
        self.last_ch = 1280
        self.head = conv_bn_act(in_ch, self.last_ch, k=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_ch, num_classes))
        self.feature_channels = {"mid": int(96 * width_mult), "final": self.last_ch}
        _init_weights(self)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        mid = self.features_low(x)
        final = self.head(self.features_high(mid))
        logits = self.classifier(self.pool(final).flatten(1))
        if return_features:
            return logits, {"mid": mid, "final": final}
        return logits


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
