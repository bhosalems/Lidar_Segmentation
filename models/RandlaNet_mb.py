import torch.nn as nn
import sklearn
from sklearn.neighbors import NearestNeighbors

class LocSE(nn.Module):
    def __init__(self, k):
        self.k = k
        self.knn = NearestNeighbors(k)

    def forward(self, coords, features):

        self.knn.fit(coords)
        knn_dist, knn_points = self.knn.kneighbors(coords,return_distance=True)

        r = []
        for i in range(knn_points.shape[0]):
            for j in range(knn_points[i].shape[0]):
                pos_enc = coords[i].tolist() + coords[knn_points[i][j]].tolist() + \
                          (coords[i] - coords[knn_points[i][j]]).tolist() + [knn_dist[i][j]]
                r.append(pos_enc)
        #send r to shared MLP
        #append its output with features
        return


# Shared MLP is implemented with 2D convolutions with kernel size 1*1. In shared MLP weights from all the input to the
# node in the MLP are all shared. Convolution does exactly that, but since we are not interested in looking at
# information of neighbors (and in most cases on unordered point-clouds there's no structure to the data and you can not
# naively convovle) we take the 1*1 kernel size which acts as a node in linear layer.
class Shared_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sz, stride=1, batch_norm=False, activation=None):
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sz,
                              stride=stride)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        """
        Input is the input tensor (could be either intermediate features of the point cloud or the point features at the
         start)
        :param x: x is the tensor input of size batch size * Channels * Length dimension
        :return: returns tensor output of size (N, C_out, L_Out).
        """
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x