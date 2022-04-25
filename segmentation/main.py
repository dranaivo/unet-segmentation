# import libraries
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import tqdm

from model import UNet
from data import DatasetCityscapes
from metrics import AccuracyMetric
from engine import Engine
import numpy as np

from PIL import Image
import random

import glob
from os.path import join

from PIL import ImageFile
from sklearn.metrics import confusion_matrix

import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CLI
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--path_to_dataset',
                        default="./unicityscape")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
    parser.add_argument('--random_seed', default=0, help='Set to ensure repeatability')
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--input_nc', default=3, type=int, help='Input depth')
    parser.add_argument('--output_nc', default=19, type=int, help='Number of semantic classes')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of workers')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--print_metrics_iter', default=1, type=int, help='Frequency of evaluation')
    parser.add_argument('--total_epochs', default=100, type=int)
    parser.add_argument('--save_epoch', default=2, type=int, help='Frequency of weight saving')
    parser.add_argument('--save', action='store_true', help='Save model weights')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--path_to_checkpoints', default="./unicityscape")

    return parser.parse_args()

# System configurations
def system_setup(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.manual_seed(args.random_seed)


def main():
    args = parse_arguments()
    system_setup(args)

    # Create network
    network = UNet(args.input_nc, args.output_nc)  # input channels, number of output classes
    print(network)

    # Create optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=250)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)

    engine = Engine(args, network, loss_fn, optimizer)
    # Training function
    if args.train:
        engine.train()
        # if args.resume:
        #     checkpoint = torch.load(glob.glob(join(args.path_to_checkpoints, "*latest*.pt"))[0])
        #     network.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optim_state_dict'])
        #     epoch = checkpoint['epoch']
        #     # loss = checkpoint['loss']

        # if args.cuda:
        #     print("Using cuda device.")
        #     network = network.to(args.cuda_device)

        # # dataloader
        # shuffle = True
        # phase = 'train'
        # data_transform = transforms.Compose([
        #     transforms.ToTensor(),  # divides float version by 255
        #     # transforms.RandomCrop(opt.image_size)
        # ])
        # set_dataloader = DatasetCityscapes(opt=args, phase=phase, data_transform=data_transform)
        # train_dataloader = torch.utils.data.DataLoader(set_dataloader, batch_size=args.batch_size, shuffle=shuffle,
        #                                           num_workers=args.n_threads, drop_last=True)

        # train_size = len(train_dataloader)
        # print(train_size)
        # network.train()

        # progress_bar = tqdm.tqdm(range(len(train_dataloader)))
        # # Initialize metrics
        # global_cm = np.zeros((args.output_nc, args.output_nc))
        # accuracy_metric = AccuracyMetric(global_cm=global_cm)
        # progress_bar.set_description('[Training]')
        # try:
        #     for epoch in range(args.total_epochs):
        #         data_iter = iter(train_dataloader)
        #         progress_bar = tqdm.tqdm(range(len(train_dataloader)))
        #         total_iter = 0
        #         for _ in progress_bar:
        #             # train_step
        #             total_iter += 1
        #             input, target = data_iter.next()
        #             if args.cuda:
        #                 input = input.to(args.cuda_device)
        #                 target = target.to(args.cuda_device)

        #             pred = network(input)
        #             loss = loss_fn(pred, target)

        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()

        #             # Show semantic segmentation metrics
        #             if total_iter % args.print_metrics_iter == 0:
        #                 with torch.no_grad():
        #                     target_ = target.cpu().numpy()
        #                     pred_ = np.argmax(pred.cpu().numpy(), axis=1)
        #                     accuracy_metric.update_values(target_, pred_, list(range(args.output_nc)))
        #                     overall_acc, average_acc, average_iou = accuracy_metric.get_values()
        #                     message = '>>> Epoch[{}/{}]({}/{}) {}: {:.4f} {}: {:.4f} {}: {:.4f} {}: {:.4f} '.format(
        #                         epoch, args.total_epochs, total_iter, len(train_dataloader), 'loss', loss.cpu().numpy(),
        #                         'OvAcc', overall_acc, 'AvAcc', average_acc, 'AvIOU', average_iou)
        #                     print(message)
        #             #
        #         if args.save and epoch % args.save_epoch == 0:
        #             print('Saving checkpoint')
        #             filename = join(args.path_to_checkpoints, 'ckp_{}.pt'.format(epoch))
        #             torch.save({
        #                 'epoch': epoch,
        #                 'model_state_dict': network.state_dict(),
        #                 'optim_state_dict': optimizer.state_dict(),
        #             }, filename)
        #             shutil.copyfile(filename, join(args.path_to_checkpoints, 'ckp_latest.pt'))
                    
        # except KeyboardInterrupt:
        #     print('Saving checkpoint')
        #     filename = join(args.path_to_checkpoints, 'ckp_{}_interrupt.pt'.format(epoch))
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': network.state_dict(),
        #         'optim_state_dict': optimizer.state_dict(),
        #     }, filename)
        #     shutil.copyfile(filename, join(args.path_to_checkpoints, 'ckp_latest.pt'))
    else:
        # evaluation function
        with torch.no_grad():
            args.batch_size = 1

            # Load checkpoint
            checkpoint = torch.load(glob.glob(join(args.path_to_checkpoints, "*latest*.pt"))[0])
            network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            epoch = checkpoint['epoch']

            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
                # transforms.CenterCrop(opt.image_size),
            ])

            shuffle = False
            phase = 'val'
            set_dataloader = DatasetCityscapes(opt=args, phase=phase, data_transform=data_transform)
            test_dataloader = torch.utils.data.DataLoader(set_dataloader, batch_size=args.batch_size, shuffle=shuffle,
                                                           num_workers=args.n_threads, drop_last=True)
            test_size = len(test_dataloader)
            print(test_size)
            network.eval()
            progress_bar = tqdm.tqdm(range(len(test_dataloader)))
            data_iter = iter(test_dataloader)
            total_iter = 0
            global_cm = 0
            accuracy_metric = AccuracyMetric(global_cm=global_cm)
            if args.cuda:
                network.to(args.cuda_device)
            print('[Testing]')
            for _ in progress_bar:
                # test_step
                total_iter += 1
                input, target = data_iter.next()
                if args.cuda:
                    input= input.to(args.cuda_device)
                    target = target.to(args.cuda_device)

                pred = network(input)

                target_ = target.cpu().numpy()
                pred_ = np.argmax(pred.cpu().numpy(), axis=1)

                accuracy_metric.update_values(pred_, target_, list(range(args.output_nc)))

            overall_acc, average_acc, average_iou = accuracy_metric.get_values()
            message = '>>> Epoch[{}] {}: {:.4f} {}: {:.4f} {}: {:.4f} '.format(
            epoch,
            'OvAcc', overall_acc, 'AvAcc', average_acc, 'AvIOU', average_iou)
            print(message)
            

if __name__ == '__main__':
    main()
