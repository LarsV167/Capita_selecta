import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu, interpolate
from torch.nn.utils import spectral_norm
import numpy as np
l1_loss = torch.nn.L1Loss()


class Block(nn.Module):
    """
    Class for the basic convolutional building block
    """

    def __init__(self, in_ch, out_ch):
        """
        Constructor.
        :param in_ch: number of input channels to the block
        :param out_ch: number of output channels of the block
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)  # leaky ReLU
        self.bn1 = nn.BatchNorm2d(out_ch)  # batch normalisation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) #0.8 weghalen

    def forward(self, x):
        """
        Returns the output of a forward pass of the block
        :param x: the input tensor
        :return: the output tensor of the block
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)  # batch normalisation
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        return x
    

class Generator(nn.Module):
    """
    Class for the generator part of the GAN.
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8):
        """
        Constructor.
        :param chs: tuple giving the number of input channels of each block in the decoder
        """
        super().__init__()
        self.chs = chs
        self.h = h  # height of image at lowest resolution level
        self.w = w  # width of image at lowest resolution level
        self.z_dim = z_dim  # dimension of latent space
        #self.label_emb = nn.Embedding(2, 256) #What should be the values here?
        self.proj_z = nn.Linear(
            255, 255 * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, 255, self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)]
        )

        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.proj_o = nn.Sequential(
            nn.Conv2d(self.chs[-1], 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )  # output layer

    def forward(self, z, seg): #Input noise and label
        """
        Returns the output of the decoder part of the VAE
        :param x: input tensor to the generator
        """
         
        b, c, h, w = seg.size()
        x = self.proj_z(z)  # fully connected layer
        x = self.reshape(x)  # reshape to image dimension
        seg=interpolate(seg, size=(seg.size(2)//8, seg.size(3)//8), mode='bilinear')
        #emb=self.label_emb(seg.detach().type(torch.long))

        x=torch.cat((x,seg),1) #add label to input (should it be concatenate? How to concatenate them? Not sure about shapes)
        
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = self.dec_blocks[i](x)
        #x = x.view(b, -1)
        x = self.proj_o(x)  # output layer
        return x


        
def custom_model1(in_chan, out_chan):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(3,3), stride=2, padding=1)),
        nn.LeakyReLU(inplace=True)
    )

def custom_model2(in_chan, out_chan, stride=2):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(3,3), stride=stride, padding=1)),
        nn.BatchNorm2d(out_chan),
        nn.LeakyReLU(inplace=True)
    )



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #self.label_emb = nn.Embedding(2, 256) #What should be the values here?
        self.layer1 = custom_model1(32, 64)
        self.layer2 = custom_model2(64, 128)
        self.layer3 = custom_model2(128, 256,stride=1)
        #self.layer4 = custom_model2(256, 512, stride=1)
        self.inst_norm = nn.InstanceNorm2d(256)
        self.conv = spectral_norm(nn.Conv2d(256, 1, kernel_size=(3,3), padding=1)) #Should be dense layer? 
        self.out=nn.Sequential(
            self.conv,
            nn.Flatten(-1),
            nn.Sigmoid(),
        )  # output layer

        # self.out=nn.Sequential(nn.Flatten(1),nn.Linear(2*256*16*16,1),
        #     nn.Sigmoid(),
        # )  
        # output layer with dense layer ( I think this should be there instead of conv2d, however when training the
        # loss does not converge )

        # #https://www.researchgate.net/figure/Discriminator-networks-architecture-All-convolution-layers-use-zero-padding-set-to-one_fig3_330470286
    def forward(self, img, seg): #input real or fake image and label 
       # x = torch.cat((img.view(img.size(0), -1),
                          #seg.view(seg.size(0), -1)), dim=1)

        x = torch.cat((seg.detach(), img.detach()), dim=1)
        x = x.view(-1, 32, 64, 64)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = leaky_relu(self.inst_norm(x))
        #x=x.view(-1,2*256*16*16) #In case of dense layer change shape of x to fit in Linear function to output 1 channel
        x = self.out(x)
        return x


#Other implementation?
# def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
#     layers = []
#     if batch_norm:
#         # If batch_norm is true add a batch norm layer
#         conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
#         batch_norm = nn.BatchNorm2d(out_channels)
#         layers = [conv_layer, batch_norm]
#     else:
#         # If batch_norm is false just add a conv layer
#         conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
#         layers.append(conv_layer)
#     return nn.Sequential(*layers)

# class Discriminator(nn.Module):
#     def __init__(self, conv_dim=32):
#         super().__init__()
#         # Define hidden convolutional layers
#         self.input = conv(3, conv_dim, kernel_size=5, stride=2, padding=2, batch_norm=False)
#         self.conv1 = conv(conv_dim, conv_dim*2, kernel_size=5, stride=2, padding=2)
#         self.conv2 = conv(conv_dim*2, conv_dim*4, kernel_size=5, stride=2, padding=2)
#         self.conv3 = conv(conv_dim*4, conv_dim*8, kernel_size=5, stride=2, padding=2)
#         self.output = conv(conv_dim*8, 1, kernel_size=5, stride=1, padding=0, batch_norm=False)
#         # Activation function
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.02)
#     def forward(self, img,seg):
#         x=torch.cat((img,seg),dim=1)
#         x = self.leaky_relu(self.input(x))
#         x = self.leaky_relu(self.conv1(x))
#         x = self.leaky_relu(self.conv2(x))
#         x = self.leaky_relu(self.conv3(x))
#         x = torch.sigmoid(self.output(x))
#         return x

def get_noise(n_samples, z_dim, device="cpu"):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """
    Samples noise vector with reparameterization trick.
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """
    Returns KLD loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(inputs, recons, mu, logvar):
    """
    Returns VAE loss, sum of reconstruction and KLD loss
    """
    return l1_loss(inputs, recons) + kld_loss(mu, logvar)
