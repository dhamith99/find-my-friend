"""Quick checks that run without data or a GPU (used by CI in Phase 8)."""
import pytest
import torch

from src.data.dataset import eval_transform, train_transform
from src.models.factory import MODEL_NAMES, build_model, count_params


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(name):
    model = build_model(name, num_classes=4, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


@pytest.mark.parametrize("name", ["mobilenet_v3_small", "resnet18"])
def test_freeze_head_trains_only_classifier(name):
    model = build_model(name, num_classes=4, pretrained=False, freeze="head")
    total, trainable = count_params(model)
    assert 0 < trainable < total * 0.5


def test_transforms_output_float_normalised():
    img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    for tf in (train_transform(224), eval_transform()):
        out = tf(img)
        assert out.dtype == torch.float32
        assert out.shape == (3, 224, 224)
