import torch.nn as nn
import torch
import sklearn
# try:
#     from torch_points import knn
# except (ModuleNotFoundError, ImportError):
#     from torch_points_kernels import knn

from sklearn.neighbors import KNeighborsClassifier as knn

class LocSE(nn.Module):
    def __init__(self, k, d_out, device):
        super(LocSE, self).__init__()
        self.k = k
        self.sharedmlp = Shared_MLP(in_channels=10, out_channels=d_out, kernel_sz=1)
        self.device = device
    def forward(self, coords, features):
        """
        returns Local Spatial encoding of point cloud such that corresponding features are aware of their
        relative spatial locations
        Args:
            coords: x y z co-ordinates for input data, b_size*N*3
            features: Output from sharedMLP is given as features b_size*N*d
        Returns: returns a tensor of size #Todo
        """
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.k)
        knn_points, knn_dist = knn_output
        B, N, K = knn_points.size()
        extended_idx = knn_points.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = knn_points.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)

        spatial_enc = torch.concat((extended_coords,
                                   neighbors,
                                   extended_coords-neighbors,
                                   knn_dist.unsqueeze(-3)),dim=-3).to_device(self.device)
        mlp_sp_enc = self.sharedmlp(spatial_enc)
        return torch.concat(mlp_sp_enc,features.expand(B, -1, N, K),dim=-3)


    def Attention_pooling(self,locse_data):
        # initialization
        return
    def forward(self):
        #computing attention score :
        # sharedMLP
        # softmax

        #weighted summation
        return


# Shared MLP is implemented with 2D convolutions with kernel size 1*1. In shared MLP weights from all the input to the
# node in the MLP are all shared. Convolution does exactly that, but since we are not interested in looking at
# information of neighbors (and in most cases on unordered point-clouds there's no structure to the data and you can not
# naively convovle) we take the 1*1 kernel size which acts as a node in linear layer.
class Shared_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sz, stride=1, batch_norm=False, activation=None):
        super(Shared_MLP, self).__init__()
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
if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    d_in = 7
    cloud = 1000*torch.randn(1, 2**16, d_in).to(device)
    lse = LocSE(16, 20)
    lse(cloud[..., :3], cloud[..., 3:7] )