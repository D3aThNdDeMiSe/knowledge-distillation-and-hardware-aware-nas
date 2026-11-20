"""
SqueezeNet student (Iandola et al., 2016), v1.1 layout, adapted to 32x32 inputs.

Fire module = "squeeze" 1x1 conv (few channels) followed by an "expand" layer that
concatenates parallel 1x1 and 3x3 convs. The classifier is fully convolutional
(1x1 conv to num_classes + global average pooling) - there is no FC layer.

CIFAR adaptations:
  * stem: 3x3 conv stride 1 (instead of 3x3 stride 2) and no initial max-pool,
    max-pools are kept after fire3 and fire5 -> spatial 32 -> 16 -> 8;
  * BatchNorm after every conv (the original has none) - SqueezeNet trains
    poorly from scratch without it on CIFAR-100.

Feature taps for distillation:
  'mid'   = output of fire5 (after 2nd pool)   [B, 256, 8, 8]
  'final' = output of fire9                    [B, 512, 8, 8]
"""
import torch
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch, k, stride=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Fire(nn.Module):
    def __init__(self, in_ch, squeeze, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = conv_bn_relu(in_ch, squeeze, 1)
        self.expand1x1 = conv_bn_relu(squeeze, expand1x1, 1)
        self.expand3x3 = conv_bn_relu(squeeze, expand3x3, 3)

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        self.stem = conv_bn_relu(3, 64, 3, stride=1)
        self.stage1 = nn.Sequential(Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
                                    nn.MaxPool2d(3, 2, 1))                          # 32 -> 16
        self.stage2 = nn.Sequential(Fire(128, 32, 128, 128), Fire(256, 32, 128, 128),
                                    nn.MaxPool2d(3, 2, 1))                          # 16 -> 8
        self.stage3 = nn.Sequential(Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                                    Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        # Fully-convolutional classifier (as in the paper).
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                                        nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.feature_channels = {"mid": 256, "final": 512}
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Final conv: small normal init (as in the reference paper).
        nn.init.normal_(self.classifier[1].weight, 0.0, 0.01)

    def forward(self, x, return_features=False):
        x = self.stage1(self.stem(x))
        mid = self.stage2(x)
        final = self.stage3(mid)
        logits = self.classifier(final).flatten(1)
        if return_features:
            return logits, {"mid": mid, "final": final}
        return logits
