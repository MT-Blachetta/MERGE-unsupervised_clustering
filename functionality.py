#import argparse
#import os
import torch
import torchvision
import numpy as np
import collections
from torch._six import string_classes
int_classes = int
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
import torchvision.transforms as transforms
from transformation import Augment, Cutout
from models import load_backbone_model, transfer_multihead_model, load_spice_model

class wrapped_resnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self,x):
        return self.backbone(x).flatten(start_dim=1)

# Custom collate function 
def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.abc.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0} # in den key namen darf sich 'idx' nicht als substring befinden
        return batch_modified

    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))


def get_augmentation(p):

    """------------- Augmentation & Transformation --------------------"""

    augmentation_method = p['augmentation_strategy']
    augmentation_type = p['augmentation_type']

    if augmentation_type == 'scan': # <return_type> torch.Tensor

        if p['augmentation_strategy'] == 'standard':
            # Standard augmentation strategy
            train_transformation = transforms.Compose([
                transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**p['augmentation_kwargs']['normalize'])
            ])
        
        elif p['augmentation_strategy'] == 'simclr':
            # Augmentation strategy from the SimCLR paper
            train_transformation = transforms.Compose([
                transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
                ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
                transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
                transforms.ToTensor(),
                transforms.Normalize(**p['augmentation_kwargs']['normalize'])
            ])
        
        elif p['augmentation_strategy'] == 'ours':
            # Augmentation strategy from our paper 
            train_transformation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
                Augment(p['augmentation_kwargs']['num_strong_augs']),
                transforms.ToTensor(),
                transforms.Normalize(**p['augmentation_kwargs']['normalize']),
                Cutout(
                    n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                    length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                    random = p['augmentation_kwargs']['cutout_kwargs']['random'])])

            print(' selected augmentation_type scan & augmentation_strategy ours ')
        
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


    else: # <input_type> PIL.Image
        aug_args = p['augmentation_kwargs'] # <return_type> list[]
        # get_train_transformations(p):
        if augmentation_method == 'standard':
                # Standard augmentation strategy
            from transformation import StandardAugmentation
            train_transformation = StandardAugmentation(p)

        elif augmentation_method == 'simclr':
                # Augmentation strategy from the SimCLR paper
            from transformation import SimclrAugmentation
            train_transformation = SimclrAugmentation(p)

        elif augmentation_method == 'scan': # 'ours' -> 'scan'
                # Augmentation strategy from our paper 
            from transformation import ScanAugmentation
            train_transformation = ScanAugmentation(p)

        elif augmentation_method == 'random':
            from transformation import RandAugmentation
            train_transformation = RandAugmentation(aug_args)

        elif augmentation_method == 'moco':
            from transformation import MocoAugmentations
            train_transformation = MocoAugmentations(aug_args)
        
        elif augmentation_method == 'barlow':
            from transformation import BarlowtwinsAugmentations
            train_transformation = BarlowtwinsAugmentations(aug_args)

        elif augmentation_method == 'multicrop':
            from transformation import MultiCropAugmentation
            train_transformation = MultiCropAugmentation(aug_args)
            
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


    return train_transformation

#train_transformations = get_train_transformations(p)


def get_train_dataloader(p,train_transformation):

    """ --------------- TRAINING: Select Dataset -------------------- """
    # dataset:
    from datasets import STL10_trainNtest
    #
    train_split = p['train_split']
    dataset_type = p['dataset_type']

    if dataset_type == 'scan': # <return_type> dict{'image': torch.Tensor,'target': int}

        if p['train_db_name'] == 'cifar-10':
            from datasets import CIFAR10
            dataset = CIFAR10(train=True, transform=train_transformation, download=True)
            #eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'cifar-20':
            from datasets import CIFAR20
            dataset = CIFAR20(train=True, transform=train_transformation, download=True)
            #eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'stl-10':
            from datasets import STL10

            if train_split == 'train':
                dataset = STL10(split='train', transform=train_transformation, download=False)
            elif train_split == 'test':
                dataset = STL10(split='test', transform=train_transformation, download=False)
            elif train_split == 'both':
                from datasets import STL10_eval
                dataset = STL10_eval(path='/space/blachetta/data',aug=train_transformation)
            elif train_split == 'unlabeled':
                dataset = STL10(split='train+unlabeled',transform=train_transformation, download=False)
            else: raise ValueError('Invalid stl10 split')

        elif p['train_db_name'] == 'imagenet':
            from datasets import ImageNet
            dataset = ImageNet(split='train', transform=train_transformation)

        elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
            from datasets import ImageNetSubset
            subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
            #dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=val_transformations)

        else:
            raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
        
        # Wrap into other dataset (__getitem__ changes)
        if p['to_augmented_dataset']: # Dataset returns an image and an augmentation of that image.
            from datasets import AugmentedDataset
            dataset = AugmentedDataset(dataset)

        if p['to_neighbors_dataset']: # Dataset returns an image and one of its nearest neighbors.
            from datasets import NeighborsDataset
            indices = np.load(p['topk_neighbors_train_path'])
            dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])
            

    else: # <return_type> (PIL.Image , label)
        dataset = torchvision.datasets.STL10('/space/blachetta/data', split=train_split,transform=train_transformation, download=True)


    ### data_loader:


    if dataset_type == 'scan': # returns single elements stacked at the lowest level to a torch.Tensor

        batch_loader = torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=True)

    else:
        batch_loader = torch.utils.data.DataLoader(dataset,num_workers=p['num_workers'],batch_size=p['batch_size'],pin_memory=True,drop_last=True,shuffle=True)


    return batch_loader


def get_val_dataloader(p):
    
    # dataset:
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
            
            

    else:
        #dataset = torchvision.datasets.STL10('/space/blachetta/data', split=split,transform=train_transformation, download=True)
        eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)


    ### data_loader:


    if dataset_type == 'scan':

        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=False, shuffle=False)

    else:
    
        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True,
                drop_last=False, shuffle=False)

    return val_dataloader

""" ------------- Build MODEL ------------------"""

def get_backbone(p):

    backbone_model_ID = p['backbone']

    # get Backbone model

    if backbone_model_ID == 'lightly_resnet18':
        resnet = torchvision.models.resnet18()
        coreModel = nn.Sequential(*list(resnet.children())[:-1])
        backbone = wrapped_resnet(coreModel)
        #backbone_outdim = 512

    elif backbone_model_ID == 'clPcl': # <return_type> dict{'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}
        from models import resnet18
        res18 = resnet18()
        backbone = res18['backbone']
        #backbone_outdim = 512

    elif backbone_model_ID == 'scatnet':
        from scatnet import ScatSimCLR
        backbone = ScatSimCLR(J=p['scatnet_args']['J'], L=p['scatnet_args']['L'], input_size=tuple(p['scatnet_args']['input_size']), res_blocks=p['scatnet_args']['res_blocks'],out_dim=p['scatnet_args']['out_dim'])
        print('get scatnet backbone ')
        #backbone_outdim = p['scatnet_args']['out_dim']

    else: raise ValueError

    return backbone


def get_head_model(p,backbone):

    model_type = p['model_type']
    p['model_args']['nheads'] = p['num_heads']
    model_args = p['model_args']
    hidden_dim = p['hidden_dim']
    num_cluster = p['num_classes']
    backbone_outdim = p['feature_dim']

    if model_type == 'backbone':
        model = backbone

    elif model_type == 'contrastiveModel':
        from models import ContrastiveModel
        model = ContrastiveModel({'backbone': backbone ,'dim': backbone_outdim },p['model_args']['head_type'],p['feature_dim'])

    elif model_type == 'clusterHeads':
        from models import ClusteringModel
        model = ClusteringModel(backbone = {'backbone': backbone ,'dim': backbone_outdim } , nclusters = num_cluster , m = model_args)
        print('clusterHeads selected')

    elif model_type == 'mlpHead':
        from models import MlpHeadModel
        model = MlpHeadModel(backbone,backbone_outdim,model_args)

    elif model_type == 'twist':
        from models import TWIST
        model = TWIST(hidden_dim,num_cluster,backbone,backbone_outdim)

    elif model_type in ['spice_linearMLP','spice_batchnormMLP','spice_linearMLP_lastBatchnorm','spice_fullBatchnorm']:
        from models import Sim2Sem
        model = Sim2Sem(backbone,model_args)

    else: raise ValueError

    return model

def get_optimizer(p,model):

    """ ------------- Optimizer ---------------- """

    if p['update_cluster_head_only']: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        #assert(len(params) == 2 * p['num_heads'])
    else:
        print('optimizer gets full model parameters')
        params = model.parameters()


    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else: raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def get_criterion(p):

    """ ------------- Loss/criterion -------------- """

    loss_ID = p['loss_type']

    second_criterion = None

    if loss_ID == 'twist_loss':
        from loss import EntLoss
        first_criterion = EntLoss(p['loss_args'], p['loss_args']['lam1'], p['loss_args']['lam2'])

    elif loss_ID == 'double_loss':
        from loss import EntLoss
        from loss import SCAN_consistencyLoss
        first_criterion = EntLoss(p['loss_args'], p['loss_args']['lam1'], p['loss_args']['lam2'])
        second_criterion = SCAN_consistencyLoss()

    elif loss_ID == 'scan_loss':
        from loss import SCANLoss
        first_criterion = SCANLoss(p['entropy_weight'])

    elif loss_ID == 'pseudolabel':
        first_criterion = torch.nn.CrossEntropyLoss()
        print('@ref[criterion_retrieval]: selected criterion: ',first_criterion)

    elif loss_ID == 'scan_selflabel':
        from loss import ConfidenceBasedCE
        first_criterion = ConfidenceBasedCE(p['loss_args']['threshold'], p['loss_args']['apply_class_balancing'])

    else: raise ValueError

    return first_criterion, second_criterion


def get_train_function(train_method):

    if train_method == 'scan':
        from training import scan_train
        train_one_epoch = scan_train
        
    elif train_method == 'twist':
        from training import twist_training
        train_one_epoch = twist_training

    elif train_method == 'simclr':
        from training import simclr_train
        train_one_epoch = simclr_train

    elif train_method == 'scan_selflabel':
        from training import selflabel_train
        train_one_epoch = selflabel_train

    elif train_method == 'pseudolabel':
        from training import pseudolabel_train
        train_one_epoch = pseudolabel_train
        print('train function is pseudolabel_train')

    elif train_method == 'double':
        from training import double_training
        train_one_epoch = double_training

    elif train_method == 'multidouble':
        from training import multidouble_training
        train_one_epoch = multidouble_training

    elif train_method == 'multitwist':
        from training import multihead_twist_train
        train_one_epoch = multihead_twist_train
        
    else: raise ValueError

    return train_one_epoch

def get_dataset(p,train_transformation):
    
    train_split = p['train_split']
    dataset_type = p['dataset_type']

    if dataset_type == 'scan': # <return_type> dict{'image': torch.Tensor,'target': int}

        if p['train_db_name'] == 'cifar-10':
            from datasets import CIFAR10
            dataset = CIFAR10(train=True, transform=train_transformation, download=True)
            #eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'cifar-20':
            from datasets import CIFAR20
            dataset = CIFAR20(train=True, transform=train_transformation, download=True)
            #eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'stl-10':
            from datasets import STL10

            if train_split == 'train':
                dataset = STL10(split='train', transform=train_transformation, download=False)
            elif train_split == 'test':
                dataset = STL10(split='test', transform=train_transformation, download=False)
            elif train_split == 'both':
                from datasets import STL10_eval
                dataset = STL10_eval(path='/space/blachetta/data',aug=train_transformation)
            elif train_split == 'unlabeled':
                dataset = STL10(split='train+unlabeled',transform=train_transformation, download=False)
            else: raise ValueError('Invalid stl10 split')

        elif p['train_db_name'] == 'imagenet':
            from datasets import ImageNet
            dataset = ImageNet(split='train', transform=train_transformation)

        elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
            from datasets import ImageNetSubset
            subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
            #dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=val_transformations)

        else:
            raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    else:
        raise ValueError('not implemented error')

    print('dataset: ',type(dataset))
    return dataset

def initialize_training(p):
    
    aug_transform = get_augmentation(p)
    val_loader = get_val_dataloader(p)  
    first_criterion, second_criterion = get_criterion(p) #@ref[criterion_retrieval]
    train_one_epoch = get_train_function(p['train_method'])

    if p['setup'] == 'pseudolabel':
        dataset = get_dataset(p,aug_transform)
        backbone = get_backbone(p)
        model = get_head_model(p,backbone)
        
        if p['pretrain_type'] == 'scan':
            pretrained = torch.load(p['pretrain_path'],map_location='cpu')
            model.load_state_dict(pretrained['model'],strict=True)
            model = transfer_multihead_model(p,model,pretrained['head'])
        elif p['pretrain_type'] == 'spice':
            model = load_spice_model(model,p['pretrain_path'],p['model_type'])        
            model = transfer_multihead_model(p,model)
        else:
            pretrained = torch.load(p['pretrain_path'],map_location='cpu')
            model.load_state_dict(pretrained,strict=True)
            model = transfer_multihead_model(p,model)
        train_loader = None
        
    else:
        train_loader = get_train_dataloader(p,aug_transform)
        backbone = get_backbone(p)
        load_backbone_model(backbone,p['pretrain_path'],p['backbone'])
        model = get_head_model(p,backbone)
        dataset = None

    print('@ref[model_related]: model_type: ',type(model))


    optimizer = get_optimizer(p,model)

    components = {'train_dataloader': train_loader,
                        'criterion': first_criterion, 
                        'second_criterion': second_criterion,
                        'model': model,
                        'optimizer': optimizer,
                        'train_method': train_one_epoch,
                        'val_dataloader': val_loader, 
                        'dataset': dataset }

    return components


def initialize_evaluation(p,best_model_path=''):
    
    val_loader = get_val_dataloader(p)
    backbone = get_backbone(p)
    model = get_head_model(p,backbone)
    
    if p['train_method'] == 'scan':
        savepoint = torch.load(p['scan_model'],map_location='cpu')
        model.load_state_dict(savepoint)
    else:
        savepoint = torch.load(best_model_path,map_location='cpu')
        model.load_state_dict(savepoint)

    return model, val_loader

