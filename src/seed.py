"""
This module provides functions to set random seeds for reproducibility in PyTorch.

Functions:
- set_seed(seed=0, cudnn="normal"): Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
- set_work_init_fn(seed): Returns a worker initialization function that sets the random seed for each worker.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed=42, cudnn="slow"):
    """
    Set the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): Seed value for random number generators. Must be an integer in the range [0, 4294967295].
        cudnn (str): Option for cuDNN library. Must be one of ['benchmark', 'normal', 'slow', 'none'].

    Returns:
        RNG (torch.Generator): A torch.Generator object with the specified seed.

    Raises:
        AssertionError: If the input arguments do not meet the specified requirements.

    Notes:
        - This function sets the seed for the following libraries: random, numpy, torch (CPU and GPU), and torch.Generator.
        - The `cudnn` option affects the randomness and speed of the cuDNN library. 
          - 'benchmark': turn on CUDNN_FIND to find the fast operation, when each iteration has the same computing graph (e.g. input size and model architecture), it can speed up a bit
          - 'normal': usually used option, accuracy differs from the digit on 0.1%
          - 'slow': it slows down computation. More accurate reproducing than 'normal', especially when gpu number keep unchanged, the accuracy is almost the same.
          - 'none'： running on gpu/cpu yields the same result, but is slow.
    """

    assert type(seed) == int and seed in range(
        0, 4294967296
    ), "`seed` must be anint in [0,4294967295]"
    assert cudnn in [
        "benchmark",
        "normal",
        "none",
        "slow",
    ], "`cudnn` must be in [ 'benchmark', 'normal', 'slow', 'none' ] "

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # seed for hash() function, affects the iteration order of dicts, sets and other mappings, str(seed) int [0; 4294967295]
    random.seed(seed)  # random and transforms, seed int or float
    np.random.seed(seed)  # numpy, seed int
    torch.manual_seed(seed)  # cpu, seed int or float
    torch.cuda.manual_seed(seed)  # gpu, seed int or float
    torch.cuda.manual_seed_all(seed)  # multi-gpu, seed int or float
    RNG = torch.Generator().manual_seed(seed)  # torch.Generator, seed int or float

    # fmt: off
    if cudnn == "none":
        torch.backends.cudnn.enabled = False  
    elif cudnn == "slow":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif cudnn == "normal":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    elif cudnn == "benchmark":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    # fmt: on

    return RNG


def set_work_init_fn(seed):
    """
    Returns a worker initialization function that sets the random seed for each worker.

    Args:
    - seed (int): The random seed to use.

    Returns:
    - worker_init_fn (function): A function that takes a worker ID as input and sets the random seed for that worker.
    """

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    return worker_init_fn
