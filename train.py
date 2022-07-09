from config.config import PATH_TRAIN
from models.RandlaNet_mb import *
import os
import torch
import logging
import argparse
from data.dataset import get_data_loader
from config.config import *
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.utils import AverageMeter
from datetime import datetime



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, help='Batch size for data loader', default=1)
    parser.add_argument('--MX_SZ', type=int, help='max size of point cloud', default=16000)
    parser.add_argument('--n_scenes', type=int, help='number of images per sequence', default=80)
    parser.add_argument('--log_dir', type=str, help='path to log file', default='logs/')
    parser.add_argument('--k', type=int, help='k neighbors', default=16)
    parser.add_argument('--d_in', type=int, help='input feature dimension', default=6)
    parser.add_argument('--decimation', type=int, help='decimation value', default=4)
    # TODO Need to change the number of classes here, just added +1 to avoid the assertion error,
    # https://github.com/scaleapi/pandaset-devkit/issues/132
    parser.add_argument('--num_classes', type=int, help='number of classes in dataset', default=43)
    parser.add_argument('--device', type=str, help='cpu/cuda', default='cuda')
    parser.add_argument('--epochs', type=int, help='number of train epochs', default=2)
    return parser.parse_args()

def randla_train(PATH, args):

    pdset_train = get_data_loader(PATH, args.b_size, args.MX_SZ, args.n_scenes, False)
    model = RandLANet(args.k, args.d_in, args.decimation, args.num_classes, args.device)
    epochs = args.epochs
    log_dir = args.log_dir
    device = args.device
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()  #TODO :add class weights for handling class imbalance
    model.double().to(device)

    handlers = [logging.StreamHandler()]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, 'logs_'+str(datetime.now().strftime("%m%d%y_%H%M%S")))
    open(log_file, 'w')
    handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

    logging.info(args)
    
    with SummaryWriter(log_dir) as writer:
        for e in range(epochs):
            logging.info(f'===========Epoch {e}/{epochs}============')
            itr_loss = AverageMeter()
            ln = len(pdset_train)
            for idx, data in enumerate(pdset_train):
                data = (data[0].to(device), data[1].to(device))
                pt_cloud, pt_labels = data

                pt_labels = pt_labels.squeeze(-1)
                pt_labels = pt_labels - 1
                print(pt_labels.max(),pt_labels.min())

                opt.zero_grad()
                scores = model(pt_cloud)

                # Information on logits - https://tinyurl.com/6wp4uwwz
                pred_label = torch.distributions.utils.probs_to_logits(scores, is_binary=False)
                loss = criterion(pred_label, pt_labels)
                itr_loss.update(loss.item())

                logging.info(f'itreration: {idx}/{ln} loss : {loss.item()}')
                loss.backward()
                opt.step()
                
            writer.add_scalar(itr_loss.avg, e)
            logging.info('Epoch completed : {e}/{epochs} loss : {itr_loss.avg}')


if __name__ == "__main__":
    args = get_args()
    PATH = PATH_TRAIN  
    randla_train(PATH, args)
