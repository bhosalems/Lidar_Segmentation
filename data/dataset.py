from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PointCloud(Dataset):
{
    def __init__(self, root_dir):
        self.root_dir = root_dir
}