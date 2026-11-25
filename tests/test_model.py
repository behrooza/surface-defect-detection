import pytest
import torch
from torch import nn

from surface_defect_detection.model import build_model


@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_build_model(num_classes):
    test_model = build_model(num_classes)
    test_model.eval()  # set the model to evaluation mode
    x = torch.randn(2, 3, 200, 200)
    # create a random input tensor with batch size 2 and image size 200x200

    with torch.no_grad():
        out = test_model(x)  # forward pass
    assert out.shape == (2, num_classes), f"""Output shape should be
                                            (2, {num_classes}), but got {out.shape}"""

    assert isinstance(test_model, nn.Module), "The model should be an instance of nn.Module"
