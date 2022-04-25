'''
Training and Evaluation procedures.
'''

import glob
from os.path import join
import shutil

import numpy as np
from PIL import ImageFile
import torch
import tqdm
import torchvision.transforms as transforms

from data import DatasetCityscapes
from metrics import AccuracyMetric

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Engine():
    def __init__(self, args, network, loss_fn, optimizer):
        self.args = args
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.network = network
        if self.args.cuda:
            print("Using cuda device.")
            self.network = self.network.to(args.cuda_device)

        if args.train:
            self.network.train()
            shuffle = True
            phase = "train"
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
            ])
            self.description = "[Training]"
            self.global_cm = np.zeros((args.output_nc, args.output_nc))
        else:
            self.network.eval()
            shuffle = False
            phase = 'val'
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
            ])
            self.description = "[Testing]"
            self.global_cm = 0

        self.set_dataloader = DatasetCityscapes(opt=args, phase=phase, data_transform=data_transform)
        self.dataloader = torch.utils.data.DataLoader(self.set_dataloader, batch_size=args.batch_size, shuffle=shuffle,
                                                           num_workers=args.n_threads, drop_last=True)
        print("Dataset size : ", len(self.dataloader))
        self.progress_bar = tqdm.tqdm(range(len(self.dataloader)))
        self.progress_bar.set_description(self.description)
        self.total_epochs = args.total_epochs
        self.accuracy_metric = AccuracyMetric(global_cm=self.global_cm)

    def save_checkpoint(self, filename: str, epoch: int) -> None:
        print('Saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }, filename)
        shutil.copyfile(filename, join(self.args.path_to_checkpoints, 'ckp_latest.pt'))

    def load_checkpoint(self) -> int:
        checkpoint = torch.load(glob.glob(join(self.args.path_to_checkpoints, "*latest*.pt"))[0])
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return epoch

    def train(self):
        if self.args.resume:
            epoch = self.load_checkpoint()
        try:
            for epoch in range(self.total_epochs):
                data_iter = iter(self.dataloader)
                total_iter = 0
                for _ in self.progress_bar:
                    total_iter += 1
                    input, target = data_iter.next()
                    if self.args.cuda:
                        input = input.to(self.args.cuda_device)
                        target = target.to(self.args.cuda_device)

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
                            self.accuracy_metric.update_values(target_, pred_, list(range(self.args.output_nc)))
                            overall_acc, average_acc, average_iou = self.accuracy_metric.get_values()
                            message = '>>> Epoch[{}/{}]({}/{}) {}: {:.4f} {}: {:.4f} {}: {:.4f} {}: {:.4f} '.format(
                                epoch, self.total_epochs, total_iter, len(self.dataloader), 'loss', loss.cpu().numpy(),
                                'OvAcc', overall_acc, 'AvAcc', average_acc, 'AvIOU', average_iou)
                            print(message)

                if self.args.save and epoch % self.args.save_epoch == 0:
                    filename = join(self.args.path_to_checkpoints, 'ckp_{}.pt'.format(epoch))
                    self.save_checkpoint(filename, epoch)
                    
        except KeyboardInterrupt:
            filename = join(self.args.path_to_checkpoints, 'ckp_{}_interrupt.pt'.format(epoch))
            self.save_checkpoint(filename, epoch)

    def evaluate(self):
        with torch.no_grad():
            self.args.batch_size = 1
            epoch = self.load_checkpoint()
            data_iter = iter(self.dataloader)
            total_iter = 0
            for _ in self.progress_bar:
                total_iter += 1
                input, target = data_iter.next()
                if self.args.cuda:
                    input= input.to(self.args.cuda_device)
                    target = target.to(self.args.cuda_device)

                pred = self.network(input)
                target_ = target.cpu().numpy()
                pred_ = np.argmax(pred.cpu().numpy(), axis=1)

                self.accuracy_metric.update_values(pred_, target_, list(range(self.args.output_nc)))

            overall_acc, average_acc, average_iou = self.accuracy_metric.get_values()
            message = '>>> Epoch[{}] {}: {:.4f} {}: {:.4f} {}: {:.4f} '.format(
                epoch,
                'OvAcc', overall_acc, 'AvAcc', average_acc, 'AvIOU', average_iou)
            print(message)