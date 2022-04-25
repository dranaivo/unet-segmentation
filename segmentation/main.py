# import libraries
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from engine import Engine
from model import UNet

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

    # Network
    network = UNet(args.input_nc, args.output_nc)
    print(network)

    # Loss and Otimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=250)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)

    engine = Engine(args, network, loss_fn, optimizer)
    if args.train:
        engine.train()
    else:
        engine.evaluate()
            

if __name__ == '__main__':
    main()
