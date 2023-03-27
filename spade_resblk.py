import torch
from torch.nn import Module, Conv2d
from torch.nn.functional import relu, interpolate
from torch.nn.utils import spectral_norm
# from .spade import SPADE # instead of creating another spade.py file I put the class from that file in here
import warnings
warnings.filterwarnings("ignore")

class SPADE(Module):
    def __init__(self, args, k):
        super().__init__()
        num_filters = args.spade_filter
        kernel_size = args.spade_kernel
        self.conv = spectral_norm(Conv2d(1, num_filters, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_gamma = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_beta = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))

    def forward(self, x, seg):
        N, C, H, W = x.size()

        sum_channel = torch.sum(x.reshape(N, C, H*W), dim=-1)
        mean = sum_channel / (N*H*W)
        std = torch.sqrt((sum_channel**2 - mean**2) / (N*H*W))

        mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
        std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        x = (x - mean) / std

        seg = interpolate(seg, size=(H,W), mode='nearest')
        seg = relu(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)
        seg_beta = self.conv_beta(seg)

        x = torch.matmul(seg_gamma, x) + seg_beta

        return x


class SPADEResBlk(Module):
    def __init__(self, args, k, skip=False):
        super().__init__()
        kernel_size = args.spade_resblk_kernel
        self.skip = skip
        
        if self.skip:
            self.spade1 = SPADE(args, 2*k)
            self.conv1 = Conv2d(2*k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
            self.spade_skip = SPADE(args, 2*k)
            self.conv_skip = Conv2d(2*k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
        else:
            self.spade1 = SPADE(args, k)
            self.conv1 = Conv2d(k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)

        self.spade2 = SPADE(args, k)
        self.conv2 = Conv2d(k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
    
    def forward(self, x, seg):
        x_skip = x
        x = relu(self.spade1(x, seg))
        x = self.conv1(x)
        x = relu(self.spade2(x, seg))
        x = self.conv2(x)

        if self.skip:
            x_skip = relu(self.spade_skip(x_skip, seg))
            x_skip = self.conv_skip(x_skip)
        
        return x_skip + x