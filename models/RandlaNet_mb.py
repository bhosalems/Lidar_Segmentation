import torch.nn as nn


class LocSE(nn.Module):
    def __init__(self, args):
        
        self.mlp = nn.Linear(args.input_channel, args.input_channels)

    def forward(self, x):
        o = self.mlp(x)


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