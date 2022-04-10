import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from config import *


class Trainset(Dataset):
    def __init__(self):
        super(Trainset, self).__init__()
        self.trajset = []
        self.path = 'D:/秦傲洋/Multigrid_learning/data/CIFAR-10-images-5000/train'
        self.transform = T.Compose([T.Resize([64, 64]), T.ToTensor()])
        self.num = 0
        for root, dirpath, filelist in os.walk(self.path):
            if dirpath == []:
                for file in filelist:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = self.transform(img)
                    self.trajset.append(img)
                    self.num += 1
                    print(self.num)
        self.trajset = torch.stack(self.trajset, dim=0)

    def __getitem__(self, index):
        return self.trajset[index]
    
    def __len__(self):
        return self.num
