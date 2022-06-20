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
        self.type = type

    def __getitem__(self, idx):
        seq_no = math.floor(idx / self.num_scenes)
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

            self.sc_ptcloud_tensor = torch.tensor(self.sc_ptcloud.values)
            self.sc_semseg_tensor = torch.tensor(self.sc_semseg.values)
            return self.sc_ptcloud_tensor, self.sc_semseg
        else:
            return self.sc_ptcloud, self.sc_semseg

    def __len__(self):
        return self.len


def get_data_loader(dir, batch, num_scenes=80, to_tensor=True):
    pdset = PandaDataset(root_dir=dir, num_scenes=num_scenes, to_tensor=to_tensor)
    return DataLoader(pdset, num_workers=4, batch_size=batch)


if __name__ == '__main__':
    train_dl = get_data_loader(r'C:\Users\akumar58\Desktop\instance segmentation\pandaset_0\train', 8, 80, True)
    valid_dl = get_data_loader(r'C:\Users\akumar58\Desktop\instance segmentation\pandaset_0\test', 8, 80, True)
    for i_batch, sample_batched in enumerate(train_dl):
        print(i_batch)
    #
    # pdset = PandaDataset(r'C:\Users\akumar58\Desktop\instance segmentation\pandaset_0\train', 8)
    # for i in range(len(pdset)):
    #     pt_cloud, label = pdset[i]
