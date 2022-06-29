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
#from utils import compute_scan_dataset, compute_default_dataset

def main():

    prefix = 'spice-scatnet_5head_batchnorm'
    model_type = 'spice_batchnormMLP'
    savefile = "/home/blachm86/SPICE-main/results/stl10/scatnet_self_5head_batchnorm/checkpoint_select.pth.tar" # put here the path to the model file 
    device = 'cuda:0'

    p = {

    'val_split': 'both',
    'dataset_type': 'scan',
    'transformation_kwargs': {'crop_size': 96, 'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'train_db_name': 'stl-10',
    'to_neighbors_dataset': True, 
    'topk_neighbors_val_path': 'RESULTS/stl-10/topk/scatnet_both_topk-val-neighbors.npy',
    'num_workers': 8,
    'batch_size': 256

    }

    m_args = {'model_type': model_type, 
            'num_neurons': [128,128,10], 
            'last_activation': 'softmax', 
            'batch_norm': True,
            'last_batchnorm': False,
            'J': 2,
            'L': 16,
            'input_size': [96,96,3],
            'res_blocks': 30,
            'out_dim': 128
            }

 
    backbone = ScatSimCLR(J=m_args['J'], L=m_args['L'], input_size=tuple(m_args['input_size']), res_blocks=m_args['res_blocks'],out_dim=m_args['out_dim'])

    # get the model from file to the code
    model = construct_spice_model(m_args,backbone)
    load_spice_model(model,savefile,model_type)

    from datasets import STL10_trainNtest, STL10_eval


    val_transformations = transforms.Compose([
                                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(**p['transformation_kwargs']['normalize'])])
        
    val_split = p['val_split']
    dataset_type = p['dataset_type']

    if dataset_type == 'scan':

        if p['train_db_name'] == 'cifar-10':
            from datasets import CIFAR10
            #dataset = CIFAR10(train=True, transform=train_transformation, download=True)
            eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'cifar-20':
            from datasets import CIFAR20
            #dataset = CIFAR20(train=True, transform=train_transformation, download=True)
            eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'stl-10':
            if val_split == 'train':
                from datasets import STL10
                eval_dataset = STL10(split='train', transform=val_transformations, download=False)
            elif val_split == 'test':
                eval_dataset = STL10(split='test', transform=val_transformations, download=False)
            elif val_split == 'both':
                eval_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)
            else: raise ValueError('Invalid stl10 split')

                #print('eval_dataset:len: ',len(eval_dataset))
                #eval_dataset = STL10(split='train',transform=val_transformations,download=False)

        else:
            raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
            

        if p['to_neighbors_dataset']: # Dataset returns an image and one of its nearest neighbors.
            from datasets import NeighborsDataset
            #print(p['topk_neighbors_train_path'])
            indices = np.load(p['topk_neighbors_val_path'])
            #print(indices.shape)
            eval_dataset = NeighborsDataset(eval_dataset, indices) # , p['num_neighbors'])

        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=False, shuffle=False)
                
                

    else:
        #dataset = torchvision.datasets.STL10('/space/blachetta/data', split=split,transform=train_transformation, download=True)
        eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)

        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                                                    batch_size=p['batch_size'], pin_memory=True,
                                                    drop_last=False, shuffle=False)


    metric_data = Analysator(device,model,val_dataloader)
    metric_data.compute_kNN_statistics(100)
    metric_data.compute_real_consistency(0.5)
    metric_data.return_statistic_summary()

    #pickle.dump(metric_data, open('SPICE_EVAL/'+prefix+".pck",'wb'))
    torch.save('SPICE_EVAL/'+prefix+".torch")

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()