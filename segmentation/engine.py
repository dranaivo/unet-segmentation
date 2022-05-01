'''Training and Evaluation procedures.

The Engine class is responsible for launching a session, be it for
training or for evaluation. Each session will mainly include :
    - A model, and its specific loss function and optimizer,
    - The path to the dataset, 
    - All training / evaluation configurations (epoch, batch_size, ...)
    - Some system configurations (whether to use the gpu)
'''
import argparse
import glob
from os.path import join
import pathlib
import shutil
from typing import Union

import numpy as np
from PIL import ImageFile
import torch
import tqdm
import torchvision.transforms as transforms

from segmentation.data import DatasetCityscapes
from segmentation.metrics import AccuracyMetric

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Engine():

    def __init__(self,
                 args: argparse.ArgumentParser,
                 network: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 path_to_dataset: Union[str, pathlib.Path],
                 total_epochs: int,
                 batch_size: int,
                 path_to_checkpoints: str,
                 save_checkpoints: bool,
                 save_frequency_in_epoch: int,
                 n_threads: int,
                 use_cuda: False = False,
                 cuda_device: int = 0,
                 phase: str = "train"):
        self.args = args
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.network = network
        self.path_to_checkpoints = path_to_checkpoints
        if len(list(glob.glob(join(self.path_to_checkpoints,
                                   "*latest*.pt")))) == 0:
            self.resume = False
        else:
            self.resume = True
        self.save_checkpoints = save_checkpoints
        self.save_frequency_in_epoch = save_frequency_in_epoch
        self.cuda = use_cuda
        self.cuda_device = cuda_device
        if self.cuda:
            print("Using cuda device.")
            self.network = self.network.to(self.cuda_device)

        if phase == "train":
            self.network.train()
            shuffle = True
            # phase = "train"
            data_transform = transforms.Compose([
                transforms.ToTensor(),    # divides float version by 255
            ])
            self.description = "[Training]"
        elif phase == "val":
            self.network.eval()
            shuffle = False
            # phase = 'val'
            data_transform = transforms.Compose([
                transforms.ToTensor(),    # divides float version by 255
            ])
            self.description = "[Testing]"

        self.global_cm = np.zeros((args.output_nc, args.output_nc))

        self.set_dataloader = DatasetCityscapes(path_to_dataset=path_to_dataset,
                                                phase=phase,
                                                data_transform=data_transform)
        self.dataloader = torch.utils.data.DataLoader(self.set_dataloader,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=n_threads,
                                                      drop_last=True)
        print("Dataset size : ", len(self.dataloader))
        self.progress_bar = tqdm.tqdm(range(len(self.dataloader)))
        self.progress_bar.set_description(self.description)
        self.total_epochs = total_epochs
        self.accuracy_metric = AccuracyMetric(global_cm=self.global_cm)

    def save_checkpoint(self, filename: str, epoch: int) -> None:
        print('Saving checkpoint')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
            }, filename)
        shutil.copyfile(filename, join(self.path_to_checkpoints,
                                       'ckp_latest.pt'))

    def load_checkpoint(self) -> int:
        checkpoint = torch.load(
            glob.glob(join(self.path_to_checkpoints, "*latest*.pt"))[0])
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return epoch

    def train(self) -> None:
        '''Training session.

        For each step (or iteration) in each epoch:
            - get images and target segmaps,
            - forward, then compute the loss,
            - compute the gradients, then apply the updates,
            - compute the metrics, if applicable.
        '''
        if self.resume:
            init_epoch = self.load_checkpoint()
        else:
            init_epoch = 0
        try:
            for epoch in range(init_epoch, self.total_epochs):
                data_iter = iter(self.dataloader)
                total_iter = 0
                for _ in self.progress_bar:
                    total_iter += 1
                    input, target = data_iter.next()
                    if self.cuda:
                        input = input.to(self.cuda_device)
                        target = target.to(self.cuda_device)

                    pred = self.network(input)
                    loss = self.loss_fn(pred, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Display semantic segmentation metrics
                    if total_iter % self.args.print_metrics_iter == 0:
                        with torch.no_grad():
                            target_ = target.cpu().numpy()
                            pred_ = np.argmax(pred.cpu().numpy(), axis=1)
                            self.accuracy_metric.update_values(
                                target_, pred_,
                                list(range(self.args.output_nc)))
                            overall_acc, average_acc, average_iou = self.accuracy_metric.get_values(
                            )
                            message = '>>> Epoch[{}/{}]({}/{}) {}: {:.4f} {}: {:.4f} {}: {:.4f} {}: {:.4f} '.format(
                                epoch, self.total_epochs, total_iter,
                                len(self.dataloader), 'loss',
                                loss.cpu().numpy(), 'OvAcc', overall_acc,
                                'AvAcc', average_acc, 'AvIOU', average_iou)
                            print(message)

                if self.save_checkpoints and epoch % self.save_frequency_in_epoch == 0:
                    filename = join(self.path_to_checkpoints,
                                    'ckp_{}.pt'.format(epoch))
                    self.save_checkpoint(filename, epoch)

        except KeyboardInterrupt:
            filename = join(self.path_to_checkpoints,
                            'ckp_{}_interrupt.pt'.format(epoch))
            self.save_checkpoint(filename, epoch)

    def evaluate(self) -> None:
        '''Evaluation session.

        A checkpoint file inside of the folder path_to_checkpoints is needed to run this 
        session. This checkpoint file must follow the pattern "*latest*.pt". 
        For each step (or iteration):
            - get images and target segmaps,
            - forward, then compute the loss,
            - compute the gradients, then apply the updates,
            - compute the metrics, if applicable.                
        '''
        with torch.no_grad():
            self.batch_size = 1
            epoch = self.load_checkpoint()
            data_iter = iter(self.dataloader)
            total_iter = 0
            for _ in self.progress_bar:
                total_iter += 1
                input, target = data_iter.next()
                if self.cuda:
                    input = input.to(self.cuda_device)
                    target = target.to(self.cuda_device)

                pred = self.network(input)
                target_ = target.cpu().numpy()
                pred_ = np.argmax(pred.cpu().numpy(), axis=1)

                self.accuracy_metric.update_values(
                    pred_, target_, list(range(self.args.output_nc)))

            overall_acc, average_acc, average_iou = self.accuracy_metric.get_values(
            )
            message = '>>> Epoch[{}] {}: {:.4f} {}: {:.4f} {}: {:.4f} '.format(
                epoch, 'OvAcc', overall_acc, 'AvAcc', average_acc, 'AvIOU',
                average_iou)
            print(message)
