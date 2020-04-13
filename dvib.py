"""dvib.py

PyTorch implementation of the Deep Variational Information Bottleneck (DVIB)
based on the "Deep Variational Information Bottleneck" paper by Alexander A. 
Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy, and their original release
in Tensorflow (https://github.com/alexalemi/vib_demo).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt


class EncoderDVIB(nn.Module):
    """Encoder for the Deep Variational Information Bottleneck (DVIB).
    """
    def __init__(self, input_size, latent_dim, out_size1=1024,
                 out_size2=1024):        
        super(EncoderDVIB, self).__init__()
        # encoder network
        self.linear1 = nn.Linear(input_size, out_size1)
        self.linear2 = nn.Linear(out_size1, out_size2)
        self.encoder_mean = nn.Linear(out_size2, latent_dim)
        self.encoder_std = nn.Linear(out_size2, latent_dim)

        # network initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.encoder_mean.weight)
        nn.init.xavier_uniform_(self.encoder_std.weight)

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

        # network initialization
        nn.init.xavier_uniform_(self.linear1.weight)

    def forward(self, x):
        # forward pass through decoder
        # return F.softmax(self.linear1(x), dim=-1) # for discrete
        return self.linear1(x)    


class DVIB(nn.Module):
    """Deep Variational Information Bottleneck (DVIB).

    Arguments:
        input_size: size of input data
        latent_dim: dimension of encoder mean and std
        output_size: size of the output data
    """
    def __init__(self, input_size, latent_dim, output_size):        
        super(DVIB, self).__init__()
        # store DVIB hyperparamenters
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.beta = 1e-3
        self.prior = Normal(torch.zeros(1,latent_dim),torch.ones(1,latent_dim))

        # initialize encoder and decoder
        self.encoder = EncoderDVIB(input_size, latent_dim)
        self.decoder = DecoderDVIB(latent_dim, output_size)

        # loss function
        self.pred_loss = nn.MSELoss(reduction='mean')  # prediction component
        self.reg_loss = nn.KLDivLoss(reduction='batchmean')  # regularization

    def forward(self, x):
        # pass input through encoder
        latent = self.encoder(x)
                
        # pass latent through decoder
        output = self.decoder(latent)

        return output, latent

    def compute_loss(self, input_data, output_data, output_latent):
        """Compute DVIB loss for a pair of input and output."""
        reg = self.reg_loss(output_latent, self.prior.sample())
        pred = self.pred_loss(input_data, output_data)
        loss = pred + self.beta*reg

        return loss

    def test_simple_sinwave(self):
        """Tests DVIB on learning how to generate a single frequency sine wave.
        """
        # optimizer
        optimizer = optim.Adam(dvib.parameters())    

        # train data (1D, continuous case)
        n_samples = 1
        x_data = torch.linspace(0, 2*math.pi, input_size)
        input_data = torch.sin(x_data) + torch.rand((n_samples, input_size))*.1

        # train
        epochs = 2500
        for epoch in range(epochs):
            train(epoch, dvib, optimizer, input_data)

        # test data (1D, continuous case)
        n_samples = 1
        x_data = torch.linspace(0, 2*math.pi, input_size)
        input_data = torch.sin(x_data) + torch.rand((n_samples, input_size))*.1
        
        # predict outputs
        dvib.eval()
        output_data, output_latent = dvib(input_data)
        output_data = output_data.detach().numpy()

        # plot results
        plt.figure()
        plt.title('DVIB Example: Single Frequency Sine Wave')
        plt.plot(x_data.squeeze(), input_data.squeeze(), label='Input')
        plt.plot(x_data.squeeze(), output_data.squeeze(), label='Output')
        plt.legend()
        plt.grid()
        plt.show()

def train(epoch, model, optimizer, input_data):
    # forward pass
    output_data, output_latent = model(input_data)

    # compute loss
    loss = model.compute_loss(input_data, output_data, output_latent)

    # backpropagate and update optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print statistics
    if epoch % 100 == 0:
        print(f'Epoch {epoch} | Loss {loss.item()}')


if __name__ == "__main__":
    # data parameters
    input_size = 500
    latent_dim = 2  # in the paper, K variable
    output_size = input_size

    # create DVIB
    dvib = DVIB(input_size, latent_dim, output_size)

    # test on single frequency sine wave
    dvib.test_simple_sinwave()

    
