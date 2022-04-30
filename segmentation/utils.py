import random

import numpy as np
import torch


def system_setup(random_seed: float) -> None:
    '''System configurations'''
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
