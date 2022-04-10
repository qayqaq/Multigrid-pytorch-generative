import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import Multigrid
from data import Trainset
# from config import Config
from utils import save_images, merge_images

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--Langevin_T', type=int, default=90)
parser.add_argument('--delta', type=float, default=0.003)
parser.add_argument('--path', type=str, default='./data/cifar10')
parser.add_argument('--sample_dir', type=str, default='sample')
parser.add_argument('--epochs', type=int, default=100)
opt = parser.parse_args()


import wandb

wandb.init(project="test_multigrid_pytorch")
wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

config = wandb.config
config.batchsize = opt.batchsize
config.learning_rate = opt.learning_rate
config.beta1 = opt.beta1
config.weight_decay = opt.weight_decay
config.Langevin_T = opt.Langevin_T
config.delta = opt.delta
config.path = opt.path
config.sample_dir = opt.sample_dir
config.epochs = opt.epochs

def main(config):

    scale_list=[1,4,16,64]

    trainset = Trainset()
    trainloader = DataLoader(trainset, batch_size=config.batchsize, shuffle=True)

    model = Multigrid(config, scale_list)

    optim = {}
    for im_sz in scale_list[1:]:
        optim[im_sz] = torch.optim.Adam(model.parameters[im_sz], lr=config.learning_rate, betas=(config.beta1, 0.999), weight_decay=config.weight_decay)

    counter = 1
    for epoch in range(config.epochs):
        samples = {}
        train_images = {}
        scalar_energy_o = {}
        scalar_energy_s = {}
        train_loss = {}
        recon_loss = {}

        for i, data in enumerate(trainloader):
            # data: bs*2*3*64*64
            data = data.cuda()

            # samples[1]
            to_sz = scale_list[0]
            samples[to_sz] = model.down_sampling[to_sz](data)
            
            # compute 4/16/64 energy_descriptor for image
            for im_sz in scale_list[1:]:
                if im_sz == scale_list[-1]:
                    train_images[im_sz] = data
                else:
                    train_images[im_sz] = model.down_sampling[im_sz](data)
                scalar_energy_o[im_sz] = model.energy_descriptor[im_sz](train_images[im_sz])

            # scale 4/16/64
            from_sz = scale_list[0]
            for to_sz in scale_list[1:]:
                optim[to_sz].zero_grad()

                # Langevin sampling of images
                samples[to_sz] = model.up_sampling[to_sz](samples[from_sz])
                samples[to_sz] = model.Langevin_sampling(samples[to_sz], to_sz)
                scalar_energy_s[to_sz] = model.energy_descriptor[to_sz](samples[to_sz])

                train_loss[to_sz] = torch.subtract(torch.mean(
                    scalar_energy_s[to_sz]), torch.mean(scalar_energy_o[to_sz]))
                
                recon_loss[to_sz] = torch.mean(torch.abs(torch.subtract(
                    train_images[to_sz], samples[to_sz])))

                train_loss[to_sz].backward()
                optim[to_sz].step()
                
                from_sz = to_sz

                if np.mod(counter, 1000) == 1:
                    img = wandb.Image(merge_images(samples[to_sz].permute(0,2,3,1).cpu().detach().numpy()), caption='counter:{}'.format(counter))
                    wandb.log({"sample_img": img})
                    

            wandb.log({
                "train_loss[4]": train_loss[4],
                "train_loss[16]": train_loss[16],
                "train_loss[64]": train_loss[64],
                "recon_loss[4]": recon_loss[4],
                "recon_loss[16]": recon_loss[16],
                "recon_loss[64]": recon_loss[64]
            })
            print('Epoch: ', epoch, 'i: ', i)
            counter += 1

if __name__ == '__main__':
    main(config)
