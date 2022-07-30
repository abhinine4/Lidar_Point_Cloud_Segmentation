from modulefinder import Module
from turtle import forward
from unicodedata import decimal
import torch.nn as nn
import torch

# random_seed = 1 # or any of your favorite number 
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# import numpy as np
# np.random.seed(random_seed)
# import random
# random.seed(random_seed)
# torch.use_deterministic_algorithms(True)

try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn


# Shared MLP is implemented with 2D convolutions with kernel size 1*1. In shared MLP weights from all the input to the
# node in the MLP are all shared. Convolution does exactly that, but since we are not interested in looking at
# information of neighbors (and in most cases on unordered point-clouds there's no structure to the data and you can not
# naively convovle) we take the 1*1 kernel size which acts as a node in linear layer.
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sz=1, stride=1, batch_norm=False, activation_fn=None, transpose=False):
        super(SharedMLP, self).__init__()

        conv = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_sz,
            stride=stride,
            padding_mode='zeros'
        )       
        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sz,
        #                       stride=stride, padding_mode='zeros')
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99)
        else:
            self.bn = None
        self.activation_fn = activation_fn

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
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocSE(nn.Module):
    def __init__(self, k, d_out, device):
        super(LocSE, self).__init__()
        self.k = k
        self.dout = d_out
        self.device = device
        self.sharedmlp = SharedMLP(in_channels=10, out_channels=d_out, batch_norm=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        """
        returns Local Spatial encoding of point cloud such that corresponding features are aware of their
        relative spatial locations
        Args:
            coords: x y z co-ordinates for input data, b_size*N*3
            features: Output from sharedMLP is given as features b_size*N*d
        Returns: returns a tensor of size #Todo
        """
        knn_points, knn_dist = knn_output
        B, N, K = knn_points.size()
        extended_idx = knn_points.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)

        spatial_enc = torch.concat((extended_coords,
                                   neighbors,
                                   extended_coords-neighbors,
                                   knn_dist.unsqueeze(-3)), dim=-3).to(self.device)
        mlp_sp_enc = self.sharedmlp(spatial_enc)
        return torch.cat((mlp_sp_enc, features.expand(B, -1, N, K)), dim=-3)


class AttentionPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionPooling, self).__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)
        self.sharedmlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, 
                                    batch_norm=True, activation_fn=nn.ReLU())
        self.softmax = nn.Softmax(dim=-2)
    
    def forward(self, features_enc):
        #computing attention score :
        score = self.softmax(self.linear(features_enc.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        out = torch.sum(score * features_enc, dim=-1, keepdim=True)
        out = self.sharedmlp(out)
        return out

class FeatureAggregation(nn.Module):
    def __init__(self, k, in_channels, out_channels, device):
        super(FeatureAggregation, self).__init__()
        # print(in_channels, out_channels)
        self.k = k
        self.device = device

        self.lse1 = LocSE(k, out_channels//2, device)
        self.lse2 = LocSE(k, out_channels//2, device)

        self.attn1 = AttentionPooling(out_channels, out_channels//2)
        self.attn2 = AttentionPooling(out_channels, out_channels)

        self.mlp1 = SharedMLP(in_channels, out_channels//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(out_channels, 2*out_channels)
        self.short = SharedMLP(in_channels, 2*out_channels, batch_norm=True)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.k)
        knn_output = (knn_output[0].to(self.device), knn_output[1].to(self.device))

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.attn1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.attn2(x)
        
        x = self.mlp2(x)

        short_out = self.short(features)
        sum = x + short_out

        return self.lrelu(sum)

class RandLANet(nn.Module):
    def __init__(self, k, d_in, decimation, num_classes, device):
        super(RandLANet, self).__init__()
        self.decimation = decimation
        self.d_in = d_in
        self.start_fc = nn.Linear(d_in, 8)
        self.k = k

        # Todo Mahesh : Need to understand why batch norm needs 4 dimension.
        self.bn_start = nn.Sequential(
             nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
             nn.LeakyReLU(0.2)
        )

        self.enc = nn.ModuleList([
            FeatureAggregation(k, 8, 16, device),
            FeatureAggregation(k, 32, 64, device),
            FeatureAggregation(k, 128, 128, device),
            FeatureAggregation(k, 256, 256, device)]
        )
        self.mid_mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())
        
        decoder_args = {
                "transpose" : True,
                "batch_norm" : True,
                "activation_fn" : nn.ReLU()
        }
                
        self.dec = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_args),
            SharedMLP(512, 128, **decoder_args),
            SharedMLP(256, 32, **decoder_args),
            SharedMLP(64, 8, **decoder_args)]
        )

        self.last_fc1 = SharedMLP(8, 64, activation_fn=nn.ReLU(), batch_norm=True)
        self.last_fc2 = SharedMLP(64, 32, activation_fn=nn.ReLU(), batch_norm=True)
        self.drop = nn.Dropout()
        self.last_fc3 = SharedMLP(32, num_classes)

        self.device = device
        self.to(device)

    def forward(self, in_cloud):
        coords = in_cloud[..., :3]
        # features = in_cloud[..., 3:self.d_in].unsqueeze(-1).permute(0, 2, 1, 3)
        N = in_cloud.shape[1]
        d = 1
        decimation_ratio = self.decimation

        x = self.start_fc(in_cloud).transpose(-2, -1).unsqueeze(-1)
        x  = self.bn_start(x)

        encoding_list = []

        for enc_layer in self.enc:
             # Todo Check the dimensions of the features and the coordinates being passed tot he enc_layer.
            x = enc_layer(coords[:, :N//d, :], x)
            encoding_list.append(x.clone())
            d = d * self.decimation
            x = x[:, :, :N//d]
        
        x = self.mid_mlp(x)

        i = 3
        for mlp in self.dec:
            neighbors, _ = knn(coords[:, :N//d, :].cpu().contiguous(), 
                               coords[:, :N*decimation_ratio//d, :].cpu().contiguous(), 
                               1)
            
            # .to() is not inplace operation.
            neighbors = neighbors.to(self.device)
            # Todo We need check what exactly these operations mean in detail specifically on the image data.

            # Approximate comments of whats happening : We are upsampling the end encoder ouput which is of N/256
            # size of the original point cloud. We use nearest enighbor interpolation, which finds the nearest neighbor
            # of the point in it's N*4/256 volume and copies its own feature.
            
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)
            x_neighbors = torch.gather(x, -2, extended_neighbors)
            x = torch.cat((x_neighbors, encoding_list[i]), dim=1)

            x = mlp(x)
            i-=1

            d //= decimation_ratio
        
        x = self.last_fc1(x)
        x = self.last_fc2(x)
        x = self.drop(x)
        scores = self.last_fc3(x)

        return scores.squeeze(-1)


if __name__ == '__main__':
    import time
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    d_in = 15
    # torch.manual_seed(42)
    cloud = 1000*torch.randn(1, 2**16, d_in).to('cuda')
    # d = 4
    # lse = LocSE(16, d, 'cpu')
    features = cloud[..., 3:d_in].unsqueeze(-1).permute(0, 2, 1, 3)
    coords = cloud[..., :3]
    # ans = lse(cloud[..., :3], features)
    # attn = AttentionPooling(in_channels=2*d, out_channels=2*d)
    # ans = attn(ans)
    # f_aggr = FeatureAggregation(16, 12, 12, 'cpu')
    # out = f_aggr(coords, features)
    model = RandLANet(16, d_in, 4, 18, 'cuda')
    scores = model(cloud)
    print('x')
