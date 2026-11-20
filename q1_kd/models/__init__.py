"""Model registry for Question 1."""
from .efficientnet_lite import EfficientNetLite0
from .mobilenetv2 import MobileNetV2
from .resnet import ResNet50
from .squeezenet import SqueezeNet

MODELS = {
    "resnet50": ResNet50,
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def build_model(name, num_classes=100):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](num_classes=num_classes)
