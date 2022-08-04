from models import construct_spice_model, load_spice_model
from functionality import collate_custom
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn import cluster
import sklearn
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as transforms
from scatnet import ScatSimCLR
from evaluate import Analysator
from functionality import initialize_training, collate_custom
from utils.config import create_config
from datasets import STL10_eval
#from utils import compute_scan_dataset, compute_default_dataset
import os
from pcloud import PyCloud



def main():

    p = create_config('SCAN_train' ,'RESULTS', 'config/SCAN.yml', 'SCAN_train')

    num_cluster = p['num_classes']
    fea_dim = p['feature_dim']
    p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
    params = initialize_training(p)
    model = params['model']

    scan_save = torch.load('/home/blachm86/SCAN_train_model.pth.tar',map_location='cpu')
    itext = model.load_state_dict(scan_save['model'],strict=True)
    print('itext: ',itext)

    best_head = copy.deepcopy(model.cluster_head[scan_save['head']])
#print('best_head: ',best_head)
#torch.save(best_head.state_dict(),'scan_transfer_head.pth')


    eval_model = MLP_head_model(model.backbone,best_head)

# now get dataset with dataloader

    print('get validation dataset')
    
    # dataset:

    val_transformations = transforms.Compose([
                                    transforms.CenterCrop(96),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
    eval_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)


    val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=8,
                    batch_size=256, pin_memory=True, collate_fn=collate_custom,
                    drop_last=False, shuffle=False)

    print('compute features in Analysator')

    eval_object = Analysator('cuda:3',eval_model,val_dataloader)

    eval_object.compute_kNN_statistics(100)
    eval_object.compute_real_consistency(0.5)
    eval_object.return_statistic_summary(0)

    torch.save(eval_object,'scan_train_analysator.torch')

#---------------------------------------------------------------------------------------------


class MLP_head_model(nn.Module):
    def __init__(self,backbone,head):
        super(MLP_head_model, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self,x,forward_pass = 'default'):

        if forward_pass == 'default':
            features = self.backbone(x)
            return torch.nn.functional.softmax(self.head(features),dim=1)

        elif forward_pass == 'features':
            return self.backbone(x)

        elif forward_pass == 'head':
            return torch.nn.functional.softmax(self.head(x),dim=1)

        else: ValueError('invalid forward pass')


if __name__ == "__main__":
    main()