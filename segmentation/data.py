'''Cityscape segmentation dataset definition.

An example of dataset structure :

cityscape/
├── gtFine_trainvaltest
│   └── gtFine
│       ├── test       
│       │   └── berlin 
│       ├── train      
│       │   └── aachen 
│       └── val
│           └── frankfurt
└── leftImg8bit
    ├── test
    │   └── berlin
    ├── train
    │   └── aachen
    └── val
        └── frankfurt
'''

import argparse
import glob
from os.path import join
import random
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_img(filename: str, task: str = 'rgb') -> Image.Image:
    image = Image.open(filename)
    return image


def dataset_cityscapes_get_semantics(root: str, phase: str) -> List[str]:
    return sorted(
        glob.glob(
            join(root, 'gtFine_trainvaltest/gtFine', phase, '*/*labelIds.png')))


def dataset_cityscapes_get_rgb(root: str, phase: str) -> List[str]:
    return sorted(glob.glob(join(root, 'leftImg8bit', phase, '*/*.png')))


def get_paths_list(root: str, phase: str) -> Tuple[List[str], List[str]]:
    rgb_images = dataset_cityscapes_get_rgb(root, phase)
    return rgb_images, dataset_cityscapes_get_semantics(root, phase)


class DatasetCityscapes(data.Dataset):

    def __init__(self,
                 opt,
                 phase: str,
                 data_transform: Callable = None) -> None:
        self.phase = phase
        self.input_list, self.target_list = get_paths_list(
            opt.path_to_dataset, phase)

        if len(self.input_list) == 0:
            raise (RuntimeError("Found no images in subfolders of: " +
                                opt.path_to_dataset + "\n"))
        else:
            print("Seems like your path is ok! =) I found {} images!".format(
                len(self.input_list)))

        if len(self.target_list) == 0:
            raise (RuntimeError("Found no images in subfolders of: " +
                                opt.path_to_dataset + "\n"))
        else:
            print("Seems like your path is ok! =) I found {} masks!".format(
                len(self.target_list)))

        self.data_transform = data_transform

        self.phase = phase

        self.void_classes = [
            0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1
        ]
        self.valid_classes = [
            7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31,
            32, 33
        ]
        self.no_instances = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]

        self.ignore_index = 250
        self.ins_ignore_value = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def encode_segmap(self, mask: np.array) -> np.array:
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def transform(self, image: Image.Image,
                  mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        from torchvision.transforms import functional as TF
        # Resize
        resize = transforms.Resize(size=(520, 520))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image,
                                                      output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __getitem__(self, index):
        input_img = Image.open(self.input_list[index])
        target = Image.open(self.target_list[index])

        input_img, target = self.transform(input_img, target)
        input_tensor = self.data_transform(input_img)

        target_np = torch.LongTensor(
            self.encode_segmap(np.array(target, dtype=np.uint8)))
        return input_tensor, target_np

    def __len__(self):
        return len(self.input_list)
