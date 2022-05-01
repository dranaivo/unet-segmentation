'''
Testing if : (integration)
- Can do a "basic" training in one epoch
'''
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from segmentation.engine import Engine
from segmentation.model import UNet
from segmentation.utils import system_setup


def test_basic_training_one_epoch():

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_nc',
        default=19,
        type=int,
    )
    parser.add_argument(
        '--print_metrics_iter',
        default=1,
        type=int,
    )

    args = parser.parse_args(["--output_nc", "19", "--print_metrics_iter", "1"])

    # System
    system_setup(0)
    use_cuda = False
    cuda_device = 0

    # Network
    network = UNet(3, args.output_nc)
    path_to_checkpoints = "path_to_checkpoints"
    save_checkpoints = False
    save_frequency_in_epoch = 1

    # Loss and Otimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=250)
    optimizer = torch.optim.Adam(network.parameters(), 10e-3)

    # Dataset
    test_data_folder = Path((os.path.abspath(os.path.dirname(__file__))),
                            "test_data")
    cityscape_path = Path(test_data_folder, "cityscape")
    num_workers_for_loader = 1

    # Training configurations
    phase = "train"
    total_epochs = 1
    batch_size = 1

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

    engine.train()
