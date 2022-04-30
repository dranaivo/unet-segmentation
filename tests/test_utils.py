'''
Testing if :
- By setting a random seed, we get the same random values (for std, numpy, torch)
'''
import random

import numpy as np
import torch 

from segmentation import utils

def test_random_generation_with_fixed_seed():

    random_seed = 0
    
    utils.system_setup(random_seed)

    int_generated = random.randint(0, 10)
    int_expected = 6
    assert int_generated == int_expected, "Generated integer is not ok."

    np_array_generated = np.random.uniform(0, 1, (2, 2))
    np_array_expected = np.array([[0.5488135, 0.71518937], 
                                    [0.60276338, 0.54488318]])
    assert np.allclose(np_array_generated, np_array_expected), "Generated numpy array is not ok."

    torch_array_generated = torch.rand((2, 2))
    torch_array_expected = torch.tensor([[0.4962566, 0.7682218], 
                                        [0.08847743, 0.13203049]])
    assert torch.allclose(torch_array_generated, torch_array_expected), "Generated torch array is not ok."
