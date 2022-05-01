'''
Testing if :
- Can instantiate a DatasetCityscapes object (check if input and target list not empty)
- Target segmap encoding gives correct results
    - all elements belonging to the void classes become the ignore_index 
    - all elements belonging to the valid classes become the correct class_map
- transform instance method produce correct shapes
- In train phase, using the __getitem__ :
    - the input has correct shape
    - the target has correct shape
'''
import os
from pathlib import Path
import pytest

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from segmentation.data import DatasetCityscapes


@pytest.fixture
def path_to_dataset():
    test_data_folder = Path((os.path.abspath(os.path.dirname(__file__))),
                            "test_data")
    path = Path(test_data_folder, "cityscape")
    return path


@pytest.fixture
def data_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def test_dataset_instanciation(path_to_dataset, data_transform):
    phase = "train"

    dataset = DatasetCityscapes(path_to_dataset, phase, data_transform)


def test_encoding_segmap_filled_with_void_classes(path_to_dataset,
                                                  data_transform):
    phase = "train"
    dataset = DatasetCityscapes(path_to_dataset, phase, data_transform)
    segmap_filled_with_void_classes = np.array([
        [1, 0, 2],
        [3, 4, 5],
        [6, -1, 9],
    ],)

    encoded_segmap = dataset.encode_segmap(segmap_filled_with_void_classes)
    expected_encoded_segmap = np.full_like(encoded_segmap, dataset.ignore_index)

    assert np.array_equal(encoded_segmap, expected_encoded_segmap)


def test_encoding_segmap_filled_with_valid_classes(path_to_dataset,
                                                   data_transform):
    phase = "train"
    dataset = DatasetCityscapes(path_to_dataset, phase, data_transform)
    segmap_filled_with_valid_classes = np.array([[7, 8, 11], [12, 13, 17],
                                                 [19, 20, 21]])

    encoded_segmap = dataset.encode_segmap(segmap_filled_with_valid_classes)
    expected_encoded_segmap = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    assert np.array_equal(encoded_segmap, expected_encoded_segmap)


def test_transformed_item_shapes(path_to_dataset, data_transform):
    phase = "train"
    dataset = DatasetCityscapes(path_to_dataset, phase, data_transform)
    input_image = Image.fromarray(np.random.randint(0, 255, size=(600, 600, 3)),
                                  mode="RGB")
    input_mask = Image.fromarray(np.random.randint(0, 33, size=(600, 600)),
                                 mode="L")

    output_image, output_mask = dataset.transform(input_image, input_mask)
    expected_shape = (256, 256)

    assert output_image.size == expected_shape
    assert output_mask.size == expected_shape


def test_correct_item_shapes(path_to_dataset, data_transform):
    phase = "train"
    dataset = DatasetCityscapes(path_to_dataset, phase, data_transform)
    index = 0

    input_array, target_array = dataset.__getitem__(index)
    expected_input_shape = torch.Size([3, 256, 256])
    expected_target_shape = torch.Size([256, 256])

    assert input_array.shape == expected_input_shape
    assert target_array.shape == expected_target_shape
