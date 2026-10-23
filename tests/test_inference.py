"""Serving must preprocess images exactly like training did (no train/serve skew)."""
import numpy as np
import torch
from PIL import Image

from src.data.dataset import MEAN, STD, eval_transform
from src.data.face import crop_square
from src.inference import LoadedModel, to_tensor


def test_serving_preprocessing_matches_training():
    rng = np.random.default_rng(0)
    face = Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8))
    lm = LoadedModel("m", "1", "r", None, ["a", "b"], 224, MEAN, STD)

    served = to_tensor(face, lm)[0]
    # training path: PIL bicubic resize -> uint8 tensor -> eval_transform
    resized = face.resize((224, 224), Image.BICUBIC)
    trained = eval_transform()(torch.from_numpy(np.array(resized)).permute(2, 0, 1))

    assert served.shape == (3, 224, 224)
    assert torch.allclose(served, trained, atol=1e-5)


def test_crop_square_stays_inside_image():
    img = Image.new("RGB", (400, 300))
    crop = crop_square(img, (350, 250, 80, 80), margin=0.5)   # face near the corner
    assert crop.size[0] == crop.size[1]
    assert crop.size[0] <= 300
