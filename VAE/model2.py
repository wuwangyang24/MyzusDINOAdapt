import sys
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import eval_genlaguerre as L
import numpy as np

# function definitions 
def kld(mu, tau, d):
    return -tau*np.sqrt(np.pi/2)*L(1/2, d/2 -1, -(mu**2)/2) + (mu**2)/2

# convex optimization problem
def kld_min(tau, d):
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    dx = 5e-3
    x = np.sqrt(max(tau**2 - d, 0))
    for step in steps:
        for i in range(10000): # TODO update this to 10000
            y1 = kld(x-dx/2, tau, d)
            y2 = kld(x+dx/2, tau, d)

            grad = (y2-y1)/dx
            x -= grad*step
    return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=(6,4), stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )    
        self.fc_mu = nn.Linear(256, latent_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input):
        result = self.conv(input)
        s = result.shape
        result = torch.reshape(result, (s[0], np.prod(s[1:])))
        mu = self.fc_mu(result)
        log_var = torch.zeros_like(mu)
        z = self.reparameterize(mu, log_var)
        return [z, mu, log_var]


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=(6, 4), stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.ConvTranspose2d(16, 8,  kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.ConvTranspose2d(8, 3,   kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.SiLU(),
        )


    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        x = self.deconv(x)
        return x



class Loss(nn.Module):
    def __init__(self, latent_dim, tau):
        super(Loss, self).__init__()
        self.mu_star = kld_min(tau, latent_dim)

    def forward(self, recons, input, mu, log_var):
        mse_loss_fn = nn.MSELoss()
        recons_loss =mse_loss_fn(input, recons)
        mu_norm = torch.linalg.norm(mu, dim=1)
        kld = 1/2*torch.square(mu_norm - self.mu_star)
        kld_loss = torch.mean(kld)
        return {'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}


