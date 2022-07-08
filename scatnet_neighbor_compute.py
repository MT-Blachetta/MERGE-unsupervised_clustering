import argparse
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from ftplib import FTP
from typing import NoReturn, Dict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from kymatio.torch import Scattering2D

FLAGS = argparse.ArgumentParser(description='stl10 split')
FLAGS.add_argument('-split',help='prefix file selection')



def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ScatSimCLR(nn.Module):

    N_RES_BLOCKS = [8, 12, 16, 30, 45]

    INPLANES = {
        8: 32,
        12: 256,
        16: 256,
        30: 256,
        45: 256
    }

    def __init__(self, J: int, L: int, input_size: Tuple[int, int, int], res_blocks: int, out_dim: int):

        """
        Args:
            J: ScatNet scale parameter

            L: ScatNet rotation parameter

            input_size: input image size. It should be (H, W, C)

            res_blocks: number of ResBlocks in adaptor network

            out_dim: output dimension of the projection space

        Raises:
            ValueError: if `J` parameter is < 1

            ValueError: if `L` parameter is < 1

            ValueError: if `input_size` is incorrect shape

            ValueError: if `res_blocks` is not supported
        """

        super(ScatSimCLR, self).__init__()

        if J < 1:
            raise ValueError('Incorrect `J` parameter')

        if L < 1:
            raise ValueError('Incorrect `L` parameter')

        if len(input_size) != 3:
            raise ValueError('`input_size` parameter should be (H, W, C)')

        if res_blocks not in self.N_RES_BLOCKS:
            raise ValueError(f'Incorrect `res_blocks` parameter. Is should be in:'
                             f'[{", ".join(self.N_RES_BLOCKS)}]')

        self._J = J
        self._L = L

        self._res_blocks = res_blocks
        self._out_dim = out_dim

        # get image height, width and channels
        h, w, c = input_size
        # ScatNet is applied for each image channel separately
        self._num_scatnet_channels = c * ((L * L * J * (J - 1)) // 2 + L * J + 1)

        # max order is always 2 - maximum possible
        self._scatnet = Scattering2D(J=J, shape=(h, w), L=L, max_order=2)

        # batch size, which is applied to ScatNet features
        self._scatnet_bn = nn.BatchNorm2d(self._num_scatnet_channels)

        # pool size in adapter network
        self._pool_size = 4

        self.inplanes = self.INPLANES[self._res_blocks]
        # adapter network
        self._adapter_network = self._create_adapter_network()

        # linear layers
        num_ftrs = 128 * self._pool_size ** 2
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _create_adapter_network(self) -> nn.Module:
        ichannels = 32 if self._res_blocks == 8 else 256

        adapter_layers = [
            # initial conv
            conv3x3(self._num_scatnet_channels, ichannels),
            nn.BatchNorm2d(ichannels),
            nn.ReLU(True),
        ]

        if self._res_blocks == 8:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 64, 4),
                self._make_layer(ResBlock, 128, 4),
            ])

        elif self._res_blocks == 12:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 4),
                self._make_layer(ResBlock, 64, 4),
                self._make_layer(ResBlock, 128, 4),
            ])

        elif self._res_blocks == 16:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 6),
                self._make_layer(ResBlock, 64, 4),
                self._make_layer(ResBlock, 128, 6)
            ])

        elif self._res_blocks == 30:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 30)
            ])

        elif self._res_blocks == 45:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 15),
                self._make_layer(ResBlock, 64, 15),
                self._make_layer(ResBlock, 128, 15)
            ])

        adapter_layers.append(nn.AdaptiveAvgPool2d(self._pool_size))
        return nn.Sequential(*adapter_layers)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scatnet = self._scatnet(x).squeeze(1)

        B, C, FN, H, W = scatnet.size()
        scatnet = scatnet.view(B, C * FN, H, W)

        h = self._adapter_network(scatnet)
        h = h.view(h.size(0), -1)

        z = self.l2(F.relu(self.l1(h)))
        return z


class STL10_trainNtest(torchvision.datasets.VisionDataset):
    def __init__(self,path,aug):
        self.aug = aug
        self.train_dataset = torchvision.datasets.STL10(path, split='train', download=False, transform=None)
        self.train_len = len(self.train_dataset)
        self.test_dataset = torchvision.datasets.STL10(path, split='test', download=False, transform=None)
        self.test_len = len(self.test_dataset)

    def __len__(self):
        return self.train_len + self.test_len

    def __getitem__(self,index):
        if index >= self.train_len:
            index -= self.train_len
            img, target = self.test_dataset[index]
        else:
            img, target = self.train_dataset[index]

        imgs = self.aug(img)

        return imgs, target

# --------------------------------------------------------------------------------------


args = FLAGS.parse_args()
stl_split = args.split
prefix = 'scatnet_'+str(stl_split)

model = ScatSimCLR(J=2, L=16, input_size=(96, 96, 3), res_blocks=30, out_dim=128)
mpath = '/home/blachm86/backbone_models/scatnet.pth'
mdict = torch.load(mpath,map_location='cpu')
model.load_state_dict(mdict,strict=True) 

#-----------------
# phase03: nearest neighbors -----------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

val_transform = transforms.Compose([
                transforms.CenterCrop(96),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])


if stl_split == 'train':
    base_dataset = datasets.STL10('/space/blachetta/data', split='train', download=False, transform=val_transform)
elif stl_split == 'test':
    base_dataset = datasets.STL10('/space/blachetta/data', split='test',download=False, transform=val_transform)
elif stl_split == 'unlabeled':
    base_dataset = datasets.STL10('/space/blachetta/data', split='train+unlabeled',download=False,transform=val_transform)
elif stl_split == 'both':
    base_dataset = STL10_trainNtest('/space/blachetta/data',aug=val_transform)
else: raise ValueError('Invalid stl10 split')

datasize = len(base_dataset)

base_dataloader = torch.utils.data.DataLoader(base_dataset, num_workers=8,
                batch_size=128, pin_memory=True, drop_last=False, shuffle=False)


val_dataset = datasets.STL10('/space/blachetta/data', split='test',download=False, transform=val_transform)
val_datasize = len(val_dataset)

val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=128, pin_memory=True, drop_last=False, shuffle=False)


#-----------------------------------------------------------------------------------------------------------------------------------------

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)

        # indices =: topk nearest neighbors indices (of self.features) for each self.n samples in the memory_bank (self.features order)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch[0].cuda(non_blocking=True)
        targets = batch[1].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
            
 

#base_dataset  <-----------------
#base_dataloader <---------------
 
temperature = 0.5
num_classes = 10
outdim = 128
 
memory_bank_base = MemoryBank(datasize,outdim,num_classes,temperature)
memory_bank_val = MemoryBank(val_datasize,outdim,num_classes,temperature)

print('Fill memory bank for mining the nearest neighbors (train) ...')
fill_memory_bank(base_dataloader, model, memory_bank_base)
topk = 20
print('Mine the nearest neighbors (Top-%d)' %(topk)) 
indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
np.save('/home/blachm86/unsupervisedClustering/RESULTS/stl-10/topk/'+prefix+"_topk-train-neighbors.npy", indices)   

   
print('Fill memory bank for mining the nearest neighbors (val) ...', 'blue')
fill_memory_bank(val_dataloader, model, memory_bank_val)
topk = 5
print('Mine the nearest neighbors (Top-%d)' %(topk)) 
indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
np.save('/home/blachm86/unsupervisedClustering/RESULTS/stl-10/topk/'+prefix+"_topk-val-neighbors.npy", indices) 


# RESULTS/stl-10/pretext/ "topk-train-neighbors.npy"
# "topk-val-neighbors.npy"
"""
ftp = FTP('vmd32539.contaboserver.net','jupyter','yxcvbnm#1234')

file = open(prefix+"_topk-train-neighbors.npy",'rb')
ftp.storbinary('STOR '+prefix+"_topk-train-neighbors.npy", file)
file.close()

file = open(prefix+"_topk-val-neighbors.npy",'rb')
ftp.storbinary('STOR '+prefix+"_topk-val-neighbors.npy", file)
file.close()


ftp.quit()

"""
