"""Random seeding shared by training and inference."""
import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
