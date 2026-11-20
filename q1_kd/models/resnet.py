"""
ResNet-50 teacher (He et al., 2016) adapted to 32x32 CIFAR images.

Changes w.r.t. the ImageNet version (standard practice for CIFAR):
  * stem = single 3x3 conv, stride 1 (instead of 7x7 stride 2 + max-pool), so the
    32x32 image is not down-sampled to 8x8 before the first residual stage.
Stage layout (bottleneck blocks): [3, 4, 6, 3], widths 64/128/256/512, expansion 4.
Spatial sizes: 32 -> 32 (layer1) -> 16 (layer2) -> 8 (layer3) -> 4 (layer4).

forward(x, return_features=True) also returns intermediate feature maps used
by the feature-based distillation loss:
  'mid'   = output of layer3   [B, 1024, 8, 8]
  'final' = output of layer4   [B, 2048, 4, 4]  (before global pooling)
"""
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, width, stride):
        super().__init__()
        out_ch = width * self.expansion
        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        # Zero-init the last BN gamma so every residual block starts as identity
        # (Goyal et al., 2017) -> more stable early training.
        nn.init.zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))


class ResNet50(nn.Module):
    def __init__(self, num_classes=100, layers=(3, 4, 6, 3)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        in_ch = 64
        stages = []
        for i, (n, width) in enumerate(zip(layers, (64, 128, 256, 512))):
            blocks = []
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(Bottleneck(in_ch, width, stride))
                in_ch = width * Bottleneck.expansion
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        self.feature_channels = {"mid": 1024, "final": 2048}
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        mid = self.layer3(x)
        final = self.layer4(mid)
        logits = self.fc(self.pool(final).flatten(1))
        if return_features:
            return logits, {"mid": mid, "final": final}
        return logits
