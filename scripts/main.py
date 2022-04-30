'''
Entry point for training and evaluation.
'''

import argparse
import os

import torch
import torch.nn as nn

from segmentation.engine import Engine
from segmentation.model import UNet
from segmentation.utils import system_setup


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--path_to_dataset', default="./unicityscape")
    parser.add_argument('--cuda', action='store_true', help='Use cuda gpu device')
    parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
    parser.add_argument('--random_seed',
                        default=0,
                        help='Set to ensure repeatability')
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--input_nc', default=3, type=int, help='Input depth')
    parser.add_argument('--output_nc',
                        default=19,
                        type=int,
                        help='Number of semantic classes')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of workers')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--print_metrics_iter',
                        default=1,
                        type=int,
                        help='Frequency of evaluation')
    parser.add_argument('--total_epochs', default=100, type=int)
    parser.add_argument('--save_epoch',
                        default=2,
                        type=int,
                        help='Frequency of weight saving')
    parser.add_argument('--save',
                        action='store_true',
                        help='Save model weights')
    parser.add_argument('--path_to_checkpoints', default="./unicityscape")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # System
    system_setup(args.random_seed)
    use_cuda = args.cuda
    cuda_device = args.cuda_device

    # Network
    network = UNet(args.input_nc, args.output_nc)
    path_to_checkpoints = args.path_to_checkpoints
    os.makedirs(path_to_checkpoints, exist_ok=True)
    save_checkpoints = args.save
    save_frequency_in_epoch = args.save_epoch
    print(network)

    # Loss and Otimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=250)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)

    # Dataset
    cityscape_path = args.path_to_dataset
    num_workers_for_loader = args.n_threads

    # Training configurations
    if args.train:
        phase = "train"
    if args.test:
        phase = "val"
    total_epochs = args.total_epochs
    batch_size = args.batch_size

    engine = Engine(args,
                    network,
                    loss_fn,
                    optimizer,
                    cityscape_path,
                    total_epochs,
                    batch_size,
                    path_to_checkpoints,
                    save_checkpoints,
                    save_frequency_in_epoch,
                    num_workers_for_loader,
                    use_cuda=use_cuda,
                    cuda_device=cuda_device,
                    phase=phase)
    if args.train:
        engine.train()
    else:
        engine.evaluate()


if __name__ == '__main__':
    main()
