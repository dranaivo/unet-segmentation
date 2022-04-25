# import libraries
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from data import DatasetCityscapes
from metrics import AccuracyMetric
import numpy as np

from PIL import Image
import random

import glob
from os.path import join

from PIL import ImageFile

import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# def training_step(data_iter, network, loss_fn, optimizer, args):
#     input, target = data_iter.next()
#     if args.cuda:
#         input = input.to(args.cuda_device)
#         target = target.to(args.cuda_device)

#     pred = network(input)
#     loss = loss_fn(pred, target)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

class Engine():
    def __init__(self, args, network, loss_fn, optimizer):
        self.args = args
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if args.train:
            self.network.train()
            shuffle = True
            phase = "train"
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
            ])
            self.description = "[Training]"
        else:
            self.network.eval()
            shuffle = False
            phase = 'val'
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
                # transforms.CenterCrop(opt.image_size),
            ])
            self.description = "[Testing]"

        self.set_dataloader = DatasetCityscapes(opt=args, phase=phase, data_transform=data_transform)
        self.dataloader = torch.utils.data.DataLoader(self.set_dataloader, batch_size=args.batch_size, shuffle=shuffle,
                                                           num_workers=args.n_threads, drop_last=True)
        print("Dataset size : ", len(self.dataloader))
        self.progress_bar = tqdm.tqdm(range(len(self.dataloader)))
        self.total_epochs = args.total_epochs
        self.global_cm = 0
        self.accuracy_metric = AccuracyMetric(global_cm=self.global_cm)

    def load_checkpoint(self):
        checkpoint = torch.load(glob.glob(join(self.args.path_to_checkpoints, "*latest*.pt"))[0])
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        # epoch = checkpoint['epoch']

    def train(self):
        pass
    def evaluate(self):
        pass