import random
import numpy as np
import torch

DATASET_NUM = 1000000
DATA = "chess_utility"

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
