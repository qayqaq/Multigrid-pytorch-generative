import os
import torch
import math
import numpy as np
import torch.nn as nn


class Multigrid():
    def __init__(self, config, scale_list=[1,4,16,64]):
        self.batchsize = config.batchsize
        self.weight_decay = config.weight_decay
        self.Langevin_T = config.Langevin_T
        self.delta = config.delta
        self.scale_list = scale_list
        self.down_sampling = {}
        self.up_sampling = {}
        self.energy_descriptor = {}
        self.parameters = {}

        for im_sz in scale_list[1:]:
            self.parameters[im_sz] = []

        # build model for down_sampling
        from_sz = scale_list[-1]
        for to_sz in scale_list[0:-1]:
            self.down_sampling[to_sz] = down_sampling(from_sz, to_sz)
            if to_sz == scale_list[0]:
                for im_sz in scale_list[1:]:
                    self.parameters[im_sz] += list(self.down_sampling[to_sz].parameters())
            else:
                self.parameters[to_sz] += list(self.down_sampling[to_sz].parameters())
        
        # build model for up_sampling
        from_sz = scale_list[0]
        for to_sz in scale_list[1:]:
            self.up_sampling[to_sz] = up_sampling(from_sz, to_sz)
            self.parameters[to_sz] += list(self.up_sampling[to_sz].parameters())
            from_sz = to_sz

        # build model for energy
        for im_sz in scale_list[1:]:
            self.energy_descriptor[im_sz] = descriptor_warpper(im_sz)
            self.parameters[im_sz] += list(self.energy_descriptor[im_sz].parameters())

        if torch.cuda.is_available():
            for sz in self.down_sampling.keys():
                self.down_sampling[sz] = self.down_sampling[sz].cuda()
            for sz in self.up_sampling.keys():
                self.up_sampling[sz] = self.up_sampling[sz].cuda()
            for sz in self.energy_descriptor.keys():
                self.energy_descriptor[sz] = self.energy_descriptor[sz].cuda()
                

    def Langevin_sampling(self, samples, to_sz):
        t = 0
        samples = samples.clone().detach()
        samples.requires_grad = True
        while t < self.Langevin_T:
            sample_energy = self.energy_descriptor[to_sz](samples).sum()
            grad = torch.autograd.grad(sample_energy, samples)[0]
            samples = samples + 0.5 * self.delta * self.delta * grad
            samples = torch.clamp(samples, 0, 1)
            t += 1
        return samples.detach()




class up_sampling(nn.Module):
    def __init__(self, from_sz, to_sz):
        super(up_sampling, self).__init__()
        filter_size = int(to_sz / from_sz)
        self.model = nn.ConvTranspose2d(3, 3, filter_size, filter_size)

    def forward(self, x):
        out = self.model(x)
        return out

class down_sampling(nn.Module):
    def __init__(self, from_sz, to_sz):
        super(down_sampling, self).__init__()
        filter_size = int(from_sz / to_sz)
        self.model = nn.Conv2d(3, 3, filter_size, filter_size)
    
    def forward(self, x):
        out = self.model(x)
        return out

def descriptor_warpper(im_sz):
    if im_sz == 4:
        return descriptor4()
    elif im_sz == 16:
        return descriptor16()
    elif im_sz == 64:
        return descriptor64()
    else:
        print('Error!! unsupported model version {}'.format(im_sz))
        exit()

class descriptor64(nn.Module):
    def __init__(self):
        super(descriptor64, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(8192, 1)

    def forward(self, x):
        out = self.model(x)
        out = torch.reshape(out, (out.shape[0], -1))
        out = self.fc(out)
        return out

class descriptor16(nn.Module):
    def __init__(self):
        super(descriptor16, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, 5, 2, 2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(32768, 1)

    def forward(self, x):
        out = self.model(x)
        out = torch.reshape(out, (out.shape[0], -1))
        out = self.fc(out)
        return out

class descriptor4(nn.Module):
    def __init__(self):
        super(descriptor4, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, 5, 2, 2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.model(x)
        out = torch.reshape(out, (out.shape[0], -1))
        out = self.fc(out)
        return out

