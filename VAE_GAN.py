# with Leacky reluand batch normalization


import torch
import torch.nn as nn
from torch.nn import Conv2d, Module
from torch.nn.functional import interpolate, relu
from torch.nn.utils import spectral_norm
from torch.nn.functional import LeakyRelU

l1_loss = torch.nn.L1Loss()


class Args:
    def __init__(self, spade_filter=128, spade_kernel=3, spade_resblk_kernel=3, gen_input_size=256, gen_hidden_size=16384):
        self.spade_filter = spade_filter
        self.spade_kernel = spade_kernel
        self.spade_resblk_kernel = spade_resblk_kernel
        self.gen_input_size = gen_input_size
        self.gen_hidden_size = gen_hidden_size
        
        if gen_hidden_size%16 != 0:
            print("Gen hidden size not multiple of 16")

spade_filter = 64
gen_input_size = 256
gen_hidden_size = 128 * 32
args = Args(spade_filter, 3, 3, gen_input_size, gen_hidden_size)



class SPADE(Module):
    def __init__(self, args, k):
        super().__init__()
        num_filters = args.spade_filter
        kernel_size = args.spade_kernel
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)  # leaky ReLU
        self.bn = nn.BatchNorm2d(k)  # batch normalisation
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
        x = self.bn(x)

        seg = interpolate(seg, size=(H,W), mode='nearest')
        seg = self.Lrelu(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)
        seg_beta = self.conv_beta(seg)
                

        x = torch.matmul(seg_gamma, x) + seg_beta

        return x

class SPADEResBlk(Module):
    def __init__(self, args, k, skip=False):
        
        super().__init__()
        kernel_size = args.spade_resblk_kernel
        self.skip = skip
        self.Lrelu = nn.LeakyReLU(0.1, inplace=True)  # leaky ReLU
        self.bn1 = nn.BatchNorm2d(k)  # batch normalisation
        self.bn2 = nn.BatchNorm2d(k, 0.8)  # batch normalisation

        
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
        x = self.Lrelu(self.spade1(x, seg))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.Lrelu(self.spade2(x, seg))
        x = self.conv2(x)
        x = self.bn2(x)

        if self.skip:
            x_skip = self.Lrelu(self.spade_skip(x_skip, seg))
            x_skip = self.conv_skip(x_skip)
            x = self.bn1(x)
        
        return x_skip + x
    
    

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
        self.bn2 = nn.BatchNorm2d(out_ch, 0.8)

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


class Encoder(nn.Module):
    """
    Class for the encoder part of the VAE.
    """

    def __init__(self, spatial_size, z_dim=256, chs=(1, 64, 128, 256)):
        """
        Constructor.
        :param chs: tuple giving the number of input channels of each block in the encoder
        """
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2)
        # height and width of images at lowest resolution level
        _h, _w = spatial_size[0] // 2 ** (len(chs) - 1), spatial_size[1] // 2 ** (
            len(chs) - 1
        )
        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """
        Returns the list of the outputs of all the blocks in the encoder
        :param x: input image tensor
        """
    
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
        x = self.out(x)
        return torch.chunk(x, 2, dim=1)  # 2 chunks - 1 each for mu and logvar



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
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
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
        self.spade1=SPADEResBlk(args,z_dim)
        self.spade2=SPADEResBlk(args,128)
        self.spade3=SPADEResBlk(args,64)
    
    
    def forward(self, z, seg):
        """
        Returns the output of the decoder part of the VAE
        :param x: input tensor to the generator
        """ 
        
        x = self.proj_z(z)  # fully connected layer
        x = self.reshape(x)  # reshape to image dimensions
        
        down1=torch.nn.UpsamplingBilinear2d(scale_factor=1/4)
        down2=torch.nn.UpsamplingBilinear2d(scale_factor=1/2)
        
        seg1=down1(seg.type(torch.float32))
        
        x = self.upconvs[0](x)
        x=self.spade1(x,seg1)
        x = self.dec_blocks[0](x)
        x = self.upconvs[1](x) 
        
        seg2=down2(seg.type(torch.float32))

        
        x=self.spade2(x,seg2)
        x = self.dec_blocks[1](x)
        x = self.upconvs[2](x)
        
        seg = seg.type(torch.float32)
        
        x=self.spade3(x,seg)
        x = self.dec_blocks[2](x)
        
        x = self.proj_o(x)  # output layer
        
        return x
    




class VAE(nn.Module):
    """
    Class for the VAE
    """

    def __init__(
        self,
        image_size=[64, 64],
        z_dim=256,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
    ):
        """
        Constructor.
        :param enc_chs: tuple giving the number of input channels of each block in the encoder
        :param dec_chs: tuple giving the number of input channels of each block in the encoder
        """
        super().__init__()
        self.encoder = Encoder(image_size, z_dim)
        self.generator = Generator(z_dim)

    def forward(self, x,seg):
        """
        Returns the output of a forward pass of the vae
        That is, both the reconstruction and mean + logvar
        :param x: the input tensor to the encoder
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        return self.generator(latent_z,seg), mu, logvar
    

def custom_model1(in_chan, out_chan):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(3,3), stride=1, padding=1)),
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
        self.conv = spectral_norm(nn.Conv2d(256, 32, kernel_size=(3,3), padding=1)) #Should be dense layer? 
        # self.out=nn.Sequential(
        #     self.conv,
        #     nn.Sigmoid(),
        # )  # output layer

        self.out=nn.Sequential(nn.Flatten(1),nn.Linear(256*16*16,1),
            nn.Sigmoid(),
        )  
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
        x = nn.LeakyRelU(self.inst_norm(x))
        x=x.view(-1,256*16*16) #In case of dense layer change shape of x to fit in Linear function to output 1 channel

        
        x = self.out(x)
        # x=x.view(32,2,64,64)
        return x
    
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