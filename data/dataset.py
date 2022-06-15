from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandaset
import math
import gc


class PandaDataset(Dataset):
    def __init__(self, root_dir, num_scenes, transform=None, to_tensor=False):
        assert isinstance(root_dir, object)
        self.root_dir = root_dir
        self.dataset = pandaset.DataSet(self.root_dir)

        # Only get the datasets with semantic segmentation annotations
        self.sequences = self.dataset.sequences(with_semseg=True)
        self.num_sequences = len(self.sequences)
        self.num_scenes = num_scenes
        self.seq_no = -1
        self.len = self.num_sequences * self.num_scenes
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        seq_no = math.floor(idx/self.num_scenes)
        self.scene_no = idx % self.num_scenes
        if self.seq_no != seq_no:
            if self.seq_no != -1:
                del self.lidar
                del self.semseg
                gc.collect()
            self.seq_no = seq_no
            self.seq = self.dataset[self.sequences[self.seq_no]].load_lidar()
            self.seq.load_semseg()
            self.lidar = self.seq.lidar
            self.semseg = self.seq.semseg
        self.sc_semseg = self.semseg.data[self.scene_no]
        self.sc_ptcloud = self.lidar.data[self.scene_no]
        if self.to_tensor:
            # todo return the tensor instead
            return self.sc_ptcloud, self.sc_semseg
        else:
            return self.sc_ptcloud, self.sc_semseg

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pdset = PandaDataset(r'C:\Users\mahes\Desktop\Projects\Segmentation\pandaset-devkit\data\PandaSet', 80)
    for i in range(len(pdset)):
        pt_cloud, label = pdset[i]