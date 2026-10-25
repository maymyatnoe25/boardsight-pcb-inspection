import torch
from torchvision.models import resnet18

from src.cnn_gradcam import gradcam


def test_gradcam_shape_and_nonnegative():
    torch.manual_seed(0)
    m = resnet18(weights=None)
    m.fc = torch.nn.Linear(512, 1)
    cams = gradcam(m, torch.randn(2, 3, 64, 64), "layer4")
    assert cams.shape == (2, 2, 2)
    assert (cams >= 0).all()