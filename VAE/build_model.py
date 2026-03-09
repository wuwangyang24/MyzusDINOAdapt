import os
import sys
import torch
from VAEs.Tilted_BetaVAE import *
from glob import glob
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel

def build_model(config, load_checkpoint=False, checkpoint_path=None, checkpoint_path_D=None):
    # create model
    model = TiltedBetaVAE(in_channels=config.model.channel, 
                          latent_dim=config.model.latent_dim,
                          tau=config.model.tau,
                          beta=config.model.beta,
                          beta_step=config.model.beta_step,
                          image_size=config.data.image_size,
                          return_z=config.model.return_z)

    # Load checkpoint
    if 'factor' not in config.model.name:
        if load_checkpoint:
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path)
                state_dict = checkpoint['state_dict']
                state_dict = OrderedDict([k[6:], v] for k, v in state_dict.items())
                model.load_state_dict(state_dict)
            else:
                print("No checkpoint to load")
    else:
        if load_checkpoint:
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path)
                state_dict = checkpoint['state_dict']
                state_dict = OrderedDict([k[6:], v] for k, v in state_dict.items())
                model[0].load_state_dict(state_dict)
            else:
                print("No checkpoint to load for vae")

            if checkpoint_path_D:
                checkpoint = torch.load(checkpoint_path_D, map_location=torch.device(config.setting.device))
                state_dict = checkpoint['state_dict']
                state_dict = OrderedDict([k[6:], v] for k, v in state_dict.items())
                model[1].load_state_dict(state_dict)
            else:
                print("No checkpoint to load for discriminator")

        
    return model