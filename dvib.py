"""dvib.py

PyTorch implementation of the Deep Variational Information Bottleneck (DVIB)
based on the "Deep Variational Information Bottleneck" paper by Alexander A. 
Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy, and their original release
in Tensorflow (https://github.com/alexalemi/vib_demo).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class EncoderDVIB(nn.Module):
    """Encoder for the Deep Variational Information Bottleneck (DVIB).
    """
    def __init__(self, input_size, latent_dim, num_latent, out_size1=1024,
                 out_size2=1024):        
        super(EncoderDVIB, self).__init__()
        self.num_latent = num_latent
        # encoder network
        self.linear1 = nn.Linear(input_size, out_size1)
        self.linear2 = nn.Linear(out_size1, out_size2)
        self.encoder_mean = nn.Linear(out_size2, latent_dim)
        self.encoder_std = nn.Linear(out_size2, latent_dim)

    def forward(self, x):
        # forward pass through encoder
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        encoder_mean = self.encoder_mean(x)
        encoder_std = F.softplus(self.encoder_std(x))

        # sample latent based on encoder outputs
        latent_dist = Normal(encoder_mean, encoder_std)
        latent = latent_dist.sample()

        return latent


class DecoderDVIB(nn.Module):
    """Decoder for the Deep Variational Information Bottleneck (DVIB).
    """
    def __init__(self, latent_dim, output_size):        
        super(DecoderDVIB, self).__init__()
        # decoder network
        self.linear1 = nn.Linear(latent_dim, output_size)

    def forward(self, x):
        # forward pass through decoder
        return F.softmax(self.linear1(x), dim=-1)
        


class DVIB(nn.Module):
    """Deep Variational Information Bottleneck (DVIB).

    Arguments:
        input_size: size of input data
        latent_dim: dimension of encoder mean and std
        num_latent: number of samples to be sampled from encoder
        output_size: size of the output data
    """
    def __init__(self, input_size, latent_dim, num_latent, output_size):        
        super(DVIB, self).__init__()
        # store DVIB hyperparamenters
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_latent = num_latent
        self.output_size = output_size

        # initialize encoder and decoder
        self.encoder = EncoderDVIB(input_size, latent_dim, num_latent)
        self.decoder = DecoderDVIB(latent_dim, output_size)

    def forward(self, x):
        # pass input through encoder
        x = self.encoder(x)
                
        # pass latent through decoder
        output = self.decoder(x)

        return output

if __name__ == "__main__":
    # data parameters
    input_size = 5
    latent_dim = 2  # in the paper, K variable
    num_latent = 3  # number of samples from encoder
    output_size = input_size

    # create DVIB
    dvib = DVIB(input_size, latent_dim, num_latent,output_size)

    # test data
    n_samples = 3
    input_data = torch.rand((n_samples, input_size))
    print('input_data', input_data)
    print('output_data', dvib(input_data))