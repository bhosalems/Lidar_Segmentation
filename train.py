from models.RandlaNet_mb import *
import torch
import logging
import argparse
from Lidar_Segmentation.data.dataset import get_data_loader
from Lidar_Segmentation.config.config import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, help='Batch size for data loader')
    parser.add_argument('--MX_SZ', type=int, help='max size of point cloud', default=16000)
    parser.add_argument('--n_scenes', type=int, help='number of images per sequence', default=80)
    parser.add_argument('--log_file', type=str, help='path to log file', default='logs/logfile')
    parser.add_argument('--k', type=int, help='k neighbors', default=16)
    return parser.parse_args()

def randla_train(data, k):
    lse = LocSE(k)
    for i_batch, sample_batched in enumerate(data):
        coords = sample_batched[...,:3]
        features = sample_batched[...,:-3]
        lse_enc = lse(coords, features)

    # return


if __name__ == "__main__":
    args = get_args()
    pdset_train = get_data_loader(PATH_TRAIN, args.b_size, args.MX_SZ, args.n_scenes, False)
    pdset_test = get_data_loader(PATH_VALID, args.b_size, args.MX_SZ, args.n_scenes, False)
    randla_train(pdset_train, args.k)
