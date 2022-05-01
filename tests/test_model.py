'''
Testing if :
- Can instantiate a UNet object
- The input before forward has correct shape
- The output after forward has correct shape (also if forward can be done)
'''
import pytest

import torch

from segmentation.model import UNet


@pytest.fixture
def n_channels() -> int:
    return 3


@pytest.fixture
def n_classes() -> int:
    return 19


def test_unet_instanciation(n_channels, n_classes):
    UNet(n_channels, n_classes)


def test_correct_input_shape(n_channels, n_classes):

    network = UNet(n_channels, n_classes)

    input_shape = torch.Size([1, network.n_channels, 16, 16])
    expected_input_shape = torch.Size([1, n_channels, 16, 16])
    assert input_shape == expected_input_shape


def test_correct_output_shape(n_channels, n_classes):

    input_tensor = torch.rand((1, n_channels, 16, 16))
    network = UNet(n_channels, n_classes)

    network.eval()
    with torch.no_grad():
        output_tensor = network(input_tensor)

    output_shape = output_tensor.shape
    expected_output_shape = torch.Size([1, n_classes, 16, 16])
    assert output_shape == expected_output_shape
