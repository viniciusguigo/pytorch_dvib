"""dvib.py

PyTorch implementation of the Deep Variational Information Bottleneck (DVIB)
based on the "Deep Variational Information Bottleneck" paper by Alexander A. 
Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy, and their original release
in Tensorflow (https://github.com/alexalemi/vib_demo).
"""
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

        return latent, encoder_mean, encoder_std


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
        #return F.softmax(self.linear1(x), dim=-1)
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

        ## loss function
        # prediction component
        self.pred_loss = nn.MSELoss(reduction='mean')
        # regularization component computed in self.compute_loss()

    def forward(self, x):
        # pass input through encoder
        latent, latent_mean, latent_std = self.encoder(x)
                
        # pass latent through decoder
        output = self.decoder(latent)

        return output, latent, latent_mean, latent_std

    def compute_loss(self, input_data, output_data, output_latent, enc_mean=None, enc_std=None):
        """Compute DVIB loss for a pair of input and output."""
        # compute KL between encoder output and prior using pytorch dists
        enc_dist = Normal(enc_mean.detach(),enc_std.detach())
        kl_vec = torch.distributions.kl.kl_divergence(enc_dist, self.prior)
        kl_loss = kl_vec.mean()

        # compute loss
        pred = self.pred_loss(input_data, output_data)
        loss = pred + self.beta*kl_loss

        return loss, pred, kl_loss

    def test_1freq_sinewave(self):
        """Tests DVIB on learning how to generate a single frequency sine wave.
        """
        # optimizer
        optimizer = optim.Adam(dvib.parameters())    

        # train
        epochs = 2500
        loss_vals = []
        for epoch in range(epochs):
            # generate train data (1D sine wave)
            n_samples = 1
            x_data = torch.linspace(0, 2*math.pi, input_size)
            input_data = torch.sin(x_data) \
                             + torch.rand((n_samples, input_size))*.1

            # update model
            loss_val = train(epoch, dvib, optimizer, input_data)
            loss_vals.append(loss_val)

        # test data (1D sine wave)
        n_samples = 1
        x_data = torch.linspace(0, 2*math.pi, input_size)
        input_data = torch.sin(x_data) + torch.rand((n_samples, input_size))*.1
        
        # predict outputs
        dvib.eval()
        output_data, output_latent, latent_mean, latent_std = dvib(input_data)
        output_data = output_data.detach().numpy()

        ## plot results
        # numerical predictions
        plt.figure()
        plt.title('DVIB Example: Single-Frequency Sine Wave Data')
        plt.plot(x_data.squeeze(), input_data.squeeze(), label='Input')
        plt.plot(x_data.squeeze(), output_data.squeeze(), label='Output')
        plt.legend()
        plt.grid()
        # loss
        plt.figure()
        plt.title('DVIB Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.plot(loss_vals)
        plt.grid()

        plt.show()

    def test_multifreq_sinewave(self):
        """Tests DVIB on learning how to generate multiple-frequency sine wave.
        """
        # optimizer
        optimizer = optim.Adam(dvib.parameters())    

        # train data (1D sine wave)
        n_samples = 1
        x_data = torch.linspace(0, 2*math.pi, input_size)
        input_data = torch.sin(x_data) + torch.rand((n_samples, input_size))*.1

        # train
        n_freqs = 3
        epochs = 2000
        loss_vals = []
        pred_loss_vals = []
        kl_loss_vals = []
        for epoch in range(epochs):
            # generate train data (1D sine wave)
            n_samples = 1
            freq = random.randint(1, n_freqs) # generate random freq (int)
            x_data = torch.linspace(0, freq*2*math.pi, input_size)
            input_data = torch.sin(x_data) \
                             + torch.rand((n_samples, input_size))*.1

            # update model
            loss_val, pred_loss_val, kl_loss_val = train(epoch, dvib, optimizer, input_data)
            loss_vals.append(loss_val)
            pred_loss_vals.append(pred_loss_val)
            kl_loss_vals.append(kl_loss_val)

        # test data (1D sine wave)
        n_samples = 1
        freq = random.randint(1, n_freqs) # generate random freq (int)
        x_data = torch.linspace(0, freq*2*math.pi, input_size)
        input_data = torch.sin(x_data) + torch.rand((n_samples, input_size))*.1
        
        # predict outputs
        dvib.eval()
        output_data, output_latent, latent_mean, latent_std = dvib(input_data)
        output_data = output_data.detach().numpy()

        # plot results
        # numerical predictions
        plt.figure()
        plt.title('DVIB Example: Multiple-Frequency Sine Wave Data')
        plt.plot(x_data.squeeze(), input_data.squeeze(), label='Input')
        plt.plot(x_data.squeeze(), output_data.squeeze(), label='Output')
        plt.legend()
        plt.grid()

        # loss
        plt.figure()
        plt.title('DVIB Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        x_epochs = torch.arange(0, epochs)
        smooth_int = 20
        plt.plot(x_epochs[::smooth_int], loss_vals[::smooth_int], '-', label='Total Loss')
        plt.plot(x_epochs[::smooth_int], pred_loss_vals[::smooth_int], '--', label='Pred Loss')
        plt.plot(x_epochs[::smooth_int], kl_loss_vals[::smooth_int], '--', label='KL Loss (mean, unscaled)')
        plt.legend()
        plt.grid()

        # visualize latent space over multiple predictions
        if self.latent_dim == 2:
            import numpy as np
            fig, ax = plt.subplots()
            plt.title('2D Encoder Mean for each Sine Frequency (when K=2)')
            colors = ['bo', 'ro', 'go', 'yo', 'ko']
            labels = [r'$\pi$', r'$4\pi$', r'$6\pi$', r'$8\pi$', r'$10\pi$']
            for i in range(n_freqs):
                freq = i+1
                n_samples_latent = 50
                for j in range(n_samples_latent):
                    # generate input data and predictions
                    x_data = torch.linspace(0, freq*2*math.pi, input_size)
                    input_data = torch.sin(x_data) + torch.rand((n_samples, input_size))*.1
                    output_data, output_latent, latent_mean, latent_std = dvib(input_data)

                    # plot latent variables
                    latent_mean = latent_mean.detach().numpy().squeeze()
                    if j == 0:  # add label
                        plt.plot(latent_mean[0], latent_mean[1], colors[i], alpha=0.5, label=labels[i])
                    else:
                        plt.plot(latent_mean[0], latent_mean[1], colors[i], alpha=0.5)
            plt.grid()
            plt.legend()

        plt.show()

    def test_mnist(self, vis_train_data=False):
        """Tests DVIB on the MNIST dataset.
        """
        # optimizer
        optimizer = optim.Adam(dvib.parameters())    

        # train data (MNIST)
        train_data = datasets.MNIST('data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        batch_size = 64
        n_samples_train = train_data.data.shape[0]
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        train_dataiter = iter(train_dataloader)

        # visualize training data
        if vis_train_data:
            # sample images and labels from dataloader
            images, labels = train_dataiter.next()

            # display a sample of them
            plt.figure()
            plt.suptitle('Batch Sampled from MNIST Dataset')
            grid_size = math.ceil(math.sqrt(batch_size))
            for index in range(batch_size):
                plt.subplot(grid_size, grid_size, index+1)
                plt.axis('off')
                plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
            plt.show()

        # train
        epochs = 1500
        loss_vals = []
        for epoch in range(epochs):
            # sample train data
            try:
                input_data, labels = train_dataiter.next()
            except StopIteration:
                train_dataiter = iter(train_dataloader)
                input_data, labels = train_dataiter.next()
            # reshape to match DVIB
            n_sampled = input_data.shape[0]
            input_data = input_data.view(n_sampled, 28*28)

            # update model
            loss_val = train(epoch, dvib, optimizer, input_data)
            loss_vals.append(loss_val)

        # test dataloader
        test_data = datasets.MNIST('data/', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        batch_size = 5
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True)
        test_dataiter = iter(test_dataloader)

        # sample test data
        input_data, labels = test_dataiter.next()
        n_sampled = input_data.shape[0]
        input_data = input_data.view(n_sampled, 28*28)
        
        # predict outputs
        dvib.eval()
        output_data, output_latent, latent_mean, latent_std = dvib(input_data)
        output_data = output_data.detach()

        # # plot results
        # # visual predictions
        input_data = input_data.view(n_sampled, 28, 28)
        output_data = output_data.view(n_sampled, 28, 28)
        plt.figure()
        plt.suptitle('DVIB MNIST Example: Input (top) vs Predicted (bottom)')
        for index in range(batch_size):
            # plot ground truth
            plt.subplot(2, batch_size, index+1)
            plt.axis('off')
            plt.imshow(input_data[index], cmap='gray_r')
            # plot prediction
            plt.subplot(2, batch_size, index+batch_size+1)
            plt.axis('off')
            plt.imshow(output_data[index], cmap='gray_r')

        # loss
        plt.figure()
        plt.title('DVIB Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.plot(loss_vals[::10])
        plt.grid()

        plt.show()


def train(epoch, model, optimizer, input_data):
    # forward pass
    output_data, output_latent, latent_mean, latent_std = model(input_data)

    # compute loss
    loss, pred, kl_loss = model.compute_loss(input_data, output_data, output_latent, latent_mean, latent_std)

    # backpropagate and update optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print statistics
    if epoch % 100 == 0:
        print('Epoch {} | Loss {} | Pred Loss {} | KL Loss: {}'.format(
            epoch, loss.item(), pred.item(), kl_loss.item()))

    return loss.item(), pred.item(), kl_loss.item()


if __name__ == "__main__":
    # # tests DVIB on single frequency sine wave
    # input_size = 200
    # latent_dim = 256  # in the paper, K variable
    # output_size = input_size
    # dvib = DVIB(input_size, latent_dim, output_size)
    # dvib.test_1freq_sinewave()
        
    # tests DVIB on multiple frequency sine wave
    input_size = 200
    latent_dim = 2  # in the paper, K variable
    output_size = input_size
    dvib = DVIB(input_size, latent_dim, output_size)
    dvib.test_multifreq_sinewave()

    # # tests DVIB on MNIST dataset
    # input_size = 28*28
    # latent_dim = 256  # in the paper, K variable
    # output_size = input_size
    # dvib = DVIB(
    #     input_size, latent_dim, output_size)
    # dvib.test_mnist()

    
