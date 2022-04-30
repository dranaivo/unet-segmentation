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
    test_data_folder = Path((os.path.abspath(os.path.dirname(__file__))), "test_data")
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

def test_encoding_segmap_filled_with_void_classes(path_to_dataset, data_transform):
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

def test_encoding_segmap_filled_with_valid_classes(path_to_dataset, data_transform):
    phase = "train"
    dataset = DatasetCityscapes(path_to_dataset, phase, data_transform)
    segmap_filled_with_valid_classes = np.array([
        [7, 8, 11],
        [12, 13, 17],
        [19, 20, 21]
    ])

    encoded_segmap = dataset.encode_segmap(segmap_filled_with_valid_classes)
    expected_encoded_segmap = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])

    assert np.array_equal(encoded_segmap, expected_encoded_segmap)

    