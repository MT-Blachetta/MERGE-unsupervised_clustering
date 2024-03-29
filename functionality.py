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

#@Authors: Wouter Van Gansbeke, Simon Vandenhende
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


#@composed and adapted: Michael Blachetta
#@origin: Wouter Van Gansbeke, Simon Vandenhende
def get_augmentation(p,aug_method=None,aug_type=None):

    """------------- Augmentation & Transformation --------------------"""

    if aug_method is None: augmentation_method = p['augmentation_strategy']
    else: augmentation_method = aug_method

    if aug_type is None: augmentation_type = p['augmentation_type']
    else: augmentation_type = aug_type

    if augmentation_type == 'scan': # <return_type> torch.Tensor

        if p['augmentation_strategy'] == 'standard':
            # Standard augmentation strategy
            train_transformation = transforms.Compose([
                #transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.125,0.125)),
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

    elif p['augmentation_type'] == 'pseudolabel': return None

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

#@composed and adapted: Michael Blachetta
#@origin: Wouter Van Gansbeke, Simon Vandenhende
def get_train_dataloader(p,train_transformation):

    """ --------------- TRAINING: Select Dataset -------------------- """
    # dataset:
    from datasets import STL10_trainNtest
    #
    train_split = p['train_split']

# <return_type> dict{'image': torch.Tensor,'target': int}

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
            dataset = STL10(split='unlabeled',transform=train_transformation, download=False)
        elif train_split == 'train+unlabeled':
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
        


    ### data_loader:
 # returns single elements stacked at the lowest level to a torch.Tensor

    batch_loader = torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=True)

    return batch_loader


#@high modified: Michael Blachetta
#@origin: Wouter Van Gansbeke, Simon Vandenhende
def get_val_dataloader(p):
    
    # dataset:
    from datasets import STL10_trainNtest, STL10_eval


    val_transformations = transforms.Compose([
                                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    
    train_split = p['train_split']
    validation_loaders = {}



    if p['val_db_name'] == 'cifar-10':
        from datasets import CIFAR10
        #dataset = CIFAR10(train=True, transform=train_transformation, download=True)
        eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)
        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                            drop_last=False, shuffle=False)
        validation_loaders['val_loader'] = val_dataloader

    elif p['val_db_name'] == 'cifar-20':
        from datasets import CIFAR20
        #dataset = CIFAR20(train=True, transform=train_transformation, download=True)
        eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)
        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                            drop_last=False, shuffle=False)
        validation_loaders['val_loader'] = val_dataloader

    elif p['val_db_name'] == 'stl-10':
        if train_split in ['train','test']:
            from datasets import STL10
            train_dataset = STL10(split='train', transform=val_transformations, download=False)
            test_dataset = STL10(split='test', transform=val_transformations, download=False)
            train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=p['num_workers'],
                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                drop_last=False, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=p['num_workers'],
                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                drop_last=False, shuffle=False)
            validation_loaders['train_split'] = train_loader
            validation_loaders['test_split'] = test_loader

        elif train_split in ['both','unlabeled','train+unlabeled']:
            labeled_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)
            val_dataloader = torch.utils.data.DataLoader(labeled_dataset, num_workers=p['num_workers'],
                            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                            drop_last=False, shuffle=False)
            validation_loaders['val_loader'] = val_dataloader

        else: 
            raise ValueError('Invalid stl10 split')


    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
        

    #if p['to_neighbors_dataset']: # Dataset returns an image and one of its nearest neighbors.
    #    from datasets import NeighborsDataset
        #print(p['topk_neighbors_train_path'])
    #    indices = np.load(p['topk_neighbors_val_path'])
        #print(indices.shape)
    #    eval_dataset = NeighborsDataset(eval_dataset, indices) # , p['num_neighbors'])


    ### data_loader:

    return validation_loaders

""" ------------- Build MODEL ------------------"""

#@high modified: Michael Blachetta
#@origin: Wouter Van Gansbeke, Simon Vandenhende
def get_direct_valDataloader(p):
    
    # dataset:
    from datasets import STL10_trainNtest, STL10_eval
    train_split = p['train_split'] 

    val_transformations = transforms.Compose([
                                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(**p['transformation_kwargs']['normalize'])])


    if p['val_db_name'] == 'cifar-10':
        from datasets import CIFAR10
        #dataset = CIFAR10(train=True, transform=train_transformation, download=True)
        eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)
        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                            drop_last=False, shuffle=False)
        return val_dataloader

    elif p['val_db_name'] == 'cifar-20':
        from datasets import CIFAR20
        #dataset = CIFAR20(train=True, transform=train_transformation, download=True)
        eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)
        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                            drop_last=False, shuffle=False)
        return  val_dataloader

    elif p['val_db_name'] == 'stl-10':
        if train_split in ['train','test','unlabeled','train+unlabeled']:
            from datasets import STL10
            train_dataset = STL10(split=train_split, transform=val_transformations, download=False)
            train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=p['num_workers'],
                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                drop_last=False, shuffle=False) # OK

            return train_loader

        elif train_split == 'both':
            labeled_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)
            val_dataloader = torch.utils.data.DataLoader(labeled_dataset, num_workers=p['num_workers'],
                            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                            drop_last=False, shuffle=False)
            return val_dataloader

        else: 
            raise ValueError('Invalid stl10 split')


    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
        

#@author: Michael Blachetta
def get_backbone(p,secondary=False):

    if secondary:
        
        backbone_model_ID = p['fixmatch_model']['backbone']

        if backbone_model_ID == 'lightly_resnet18':
            resnet = torchvision.models.resnet18()
            coreModel = nn.Sequential(*list(resnet.children())[:-1])
            backbone = wrapped_resnet(coreModel)
            #backbone_outdim = 512

        elif backbone_model_ID in ['clPcl','resnet18']: # <return_type> dict{'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}
            from models import resnet18
            res18 = resnet18()
            backbone = res18['backbone']
            #backbone_outdim = 512

        elif backbone_model_ID in ["ResNet18","ResNet34","ResNet50"]:
            from resnet import ResNet, get_resnet
            backbone = get_resnet(backbone_model_ID)

        elif backbone_model_ID == 'scatnet':
            from scatnet import ScatSimCLR
            backbone = ScatSimCLR(J=p['fixmatch_model']['scatnet_args']['J'], L=p['fixmatch_model']['scatnet_args']['L'], input_size=tuple(p['fixmatch_model']['scatnet_args']['input_size']), res_blocks=p['fixmatch_model']['scatnet_args']['res_blocks'],out_dim=p['fixmatch_model']['scatnet_args']['out_dim'])
            print('get scatnet backbone ')
            #backbone_outdim = p['scatnet_args']['out_dim']

        else: raise ValueError

    else:
        
        backbone_model_ID = p['backbone']

        # get Backbone model

        if backbone_model_ID == 'lightly_resnet18':
            resnet = torchvision.models.resnet18()
            coreModel = nn.Sequential(*list(resnet.children())[:-1])
            backbone = wrapped_resnet(coreModel)
            #backbone_outdim = 512

        elif backbone_model_ID in ['clPcl','resnet18']: # <return_type> dict{'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}
            from models import resnet18
            res18 = resnet18()
            backbone = res18['backbone']
            #backbone_outdim = 512

        elif backbone_model_ID in ["ResNet18","ResNet34","ResNet50"]:
            from resnet import ResNet, get_resnet
            backbone = get_resnet(p['backbone'])

        elif backbone_model_ID == 'scatnet':
            from scatnet import ScatSimCLR
            backbone = ScatSimCLR(J=p['scatnet_args']['J'], L=p['scatnet_args']['L'], input_size=tuple(p['scatnet_args']['input_size']), res_blocks=p['scatnet_args']['res_blocks'],out_dim=p['scatnet_args']['out_dim'])
            print('get scatnet backbone ')
            #backbone_outdim = p['scatnet_args']['out_dim']

        else: raise ValueError

    return backbone


#@author: Michael Blachetta
def get_head_model(p,backbone,secondary=False):


    if secondary:
        
        model_type = p['fixmatch_model']['model_type']
        #hidden_dim = p['hidden_dim']
        num_cluster = p['num_classes']
        backbone_outdim = p['fixmatch_model']['feature_dim']

        if model_type == 'clusterHeads':
            from models import ClusteringModel
            model_args = p['fixmatch_model']['model_args']
            model = ClusteringModel(backbone = {'backbone': backbone ,'dim': backbone_outdim } , nclusters = num_cluster , m = model_args)
        
        elif model_type == 'mlpHead':
            from models import MlpHeadModel
            model_args = p['fixmatch_model']['model_args']
            model = MlpHeadModel(backbone,backbone_outdim,model_args)

        elif model_type in ['spice_linearMLP','spice_batchnormMLP','spice_linearMLP_lastBatchnorm','spice_fullBatchnorm']:
            from models import Sim2Sem
            model_args = p['fixmatch_model']['model_args']
            model = Sim2Sem(backbone,model_args)

        elif model_type == 'cc_network':
            from models import Network
            print('resNet Backbone dimension = ',backbone.rep_dim)
            model = Network(backbone,p['fixmatch_model']['feature_dim'],p['num_classes'])

        else: raise ValueError

    else:
        
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
        
        elif model_type == 'mlpHead':
            from models import MlpHeadModel
            model = MlpHeadModel(backbone,backbone_outdim,model_args)

        elif model_type == 'twist':
            from models import TWIST
            model = TWIST(hidden_dim,num_cluster,backbone,backbone_outdim)

        elif model_type in ['spice_linearMLP','spice_batchnormMLP','spice_linearMLP_lastBatchnorm','spice_fullBatchnorm']:
            from models import Sim2Sem
            model = Sim2Sem(backbone,model_args)

        elif model_type == 'contrastive_clustering':
            from contrastive_clustering import Network
            print('resNet Backbone dimension = ',backbone.rep_dim)
            model = Network(backbone,p['feature_dim'],p['num_classes'])

        elif model_type == 'fixmatch':
            from models import Network
            print('resNet Backbone dimension = ',backbone.rep_dim)
            model = Network(backbone,p['feature_dim'],p['num_classes'])

        else: raise ValueError

    return model



#@authors: Wouter Van Gansbeke, Simon Vandenhende
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
        
        params = model.parameters()


    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else: raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


#@author: Michael Blachetta
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
        #print('@ref[criterion_retrieval]: selected criterion: ',first_criterion)

    elif loss_ID == 'scan_selflabel':
        from loss import ConfidenceBasedCE
        first_criterion = ConfidenceBasedCE(p['loss_args']['threshold'], p['loss_args']['apply_class_balancing'])

    elif loss_ID == 'contrastive_clustering':
        from loss import InstanceLoss, ClusterLoss
        first_criterion = InstanceLoss(p['batch_size'],p['instance_temperature'])
        second_criterion = ClusterLoss(p['num_classes'],p['cluster_temperature'])

    else: raise ValueError

    return first_criterion, second_criterion


#@author: Michael Blachetta
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
        #print('train function is pseudolabel_train')

    elif train_method == 'double':
        from training import double_training
        train_one_epoch = double_training

    elif train_method == 'multidouble':
        from training import multidouble_training
        train_one_epoch = multidouble_training

    elif train_method == 'multitwist':
        from training import multihead_twist_train
        train_one_epoch = multihead_twist_train

    elif train_method == 'contrastive_clustering_stl10':
        from training import contrastive_clustering_STL10
        train_one_epoch = contrastive_clustering_STL10
        
    else: raise ValueError

    return train_one_epoch


#@composed and adapted: Michael Blachetta
#@origin: Wouter Van Gansbeke, Simon Vandenhende
def get_dataset(p,train_transformation,train=True):
    
    if train:
        _split = p['train_split']
    else:
        _split = 'both'

    # <return_type> dict{'image': torch.Tensor,'target': int}

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

        if _split == 'train':
            dataset = STL10(split='train', transform=train_transformation, download=False)
        elif _split == 'test':
            dataset = STL10(split='test', transform=train_transformation, download=False)
        elif _split == 'both':
            from datasets import STL10_eval
            dataset = STL10_eval(path='/space/blachetta/data',aug=train_transformation)
        elif _split == 'unlabeled':
            dataset = STL10(split='unlabeled',transform=train_transformation, download=False)
        elif _split == 'train+unlabeled':
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

    print('dataset: ',type(dataset))
    return dataset


#@author: Michael Blachetta
def initialize_training(p):
    
    aug_transform = get_augmentation(p)
    val_loader = get_val_dataloader(p)  
    first_criterion, second_criterion = get_criterion(p) #@ref[criterion_retrieval]
    train_one_epoch = get_train_function(p['train_method'])
    #val_transformations = transforms.Compose([
    #                        transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
    #                        transforms.ToTensor(),
    #                        transforms.Normalize(**p['transformation_kwargs']['normalize'])])

    if p['setup'] == 'pseudolabel':
        dataset = get_dataset(p,aug_transform)
        #val_dataset = get_dataset(p,val_transformations,train=False)
        backbone = get_backbone(p)
        model = get_head_model(p,backbone)
        p['augmentation_type'] = 'scan'
        strong_transform = get_augmentation(p)
        
        if p['pretrain_type'] == 'scan':
            pretrained = torch.load(p['pretrain_path'],map_location='cpu')
            model.load_state_dict(pretrained['model'],strict=True)
            model = transfer_multihead_model(p,model,pretrained['head'])
        elif p['pretrain_type'] == 'spice':
            model = load_spice_model(model,p['pretrain_path'],p['model_type'])        
            model = transfer_multihead_model(p,model)
        # elif p['pretrain_type'] == 'cc_network' # to do: implement
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
        strong_transform = None

    print('@ref[model_related]: model_type: ',type(model))


    optimizer = get_optimizer(p,model)

    components = {'train_dataloader': train_loader,
                        'criterion': first_criterion, 
                        'second_criterion': second_criterion,
                        'model': model,
                        'optimizer': optimizer,
                        'train_method': train_one_epoch,
                        'val_dataloader': val_loader, 
                        'dataset': dataset, 
                        'augmentation': strong_transform}

    return components


#@author: Michael Blachetta
def initialize_contrastive_clustering(p):

    train_transformation = get_augmentation(p)
    
    if p['train_db_name'] == 'stl-10':
        from datasets import STL10, STL10_eval
        dataset_unlabeled = STL10(split='unlabeled',transform=train_transformation, download=False)
        
        train_dataset = torchvision.datasets.STL10(
            root='/space/blachetta/data',
            split="train",
            download=True,
            transform=train_transformation,
        )
        test_dataset = torchvision.datasets.STL10(
            root='/space/blachetta/data',
            split="test",
            download=True,
            transform=train_transformation,
        )

        dataset_label = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

        unlabeled_loader = torch.utils.data.DataLoader(dataset_unlabeled, num_workers=p['num_workers'], 
                                                        batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                        drop_last=True, shuffle=True)

        labeled_loader = torch.utils.data.DataLoader(dataset_label, num_workers=p['num_workers'], 
                                                    batch_size=p['batch_size'], pin_memory=True,
                                                    drop_last=True, shuffle=True)

        dataset_loader = None

    else: 
        dataset_loader = get_train_dataloader(p,train_transformation)
        unlabeled_loader = None
        labeled_loader = None

    instance_criterion, cluster_criterion = get_criterion(p) #@ref[criterion_retrieval]
    train_one_epoch = get_train_function(p['train_method'])
    val_loader = get_val_dataloader(p)
    backbone = get_backbone(p)
    model = get_head_model(p,backbone)
    optimizer = get_optimizer(p,model)

    components = {'train_dataloader': dataset_loader,
                  'unlabeled_dataloader': unlabeled_loader,
                  'labeled_dataloader': labeled_loader,
                  'instance_criterion': instance_criterion, 
                  'cluster_criterion': cluster_criterion,
                  'model': model,
                  'optimizer': optimizer,
                  'train_method': train_one_epoch,
                  'val_dataloader': val_loader }

    return components


#@author: Michael Blachetta
def initialize_fixmatch_training(p):

    # Dataset

    dataset = get_dataset(p,None)
    val_loader = get_val_dataloader(p)

    strong_transform = get_augmentation(p)   
    weak_transform = get_augmentation(p,aug_method='standard',aug_type='scan')
    val_transform = transforms.Compose([
                            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                            transforms.ToTensor(),
                            transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    medium_transform = get_augmentation(p,aug_method='simclr',aug_type='scan')

    from datasets import pseudolabelDataset, fixmatchDataset
    unlabeled_data = fixmatchDataset(dataset,weak_transform,strong_transform)
    reliable_samples = torch.load(p['indexfile_path'],map_location='cpu')
    labeled_data = pseudolabelDataset(dataset, reliable_samples['sample_index'], reliable_samples['pseudolabel'], val_transform, medium_transform)
    base_dataloader = get_direct_valDataloader(p)

    # Model
    """
    pretrain_backbone = get_backbone(p)
    pretrain_model = get_head_model(p,pretrain_backbone)

    if p['pretrain_type'] == 'scan':
        pretrained = torch.load(p['pretrain_path'],map_location='cpu')
        pretrain_model.load_state_dict(pretrained['model'],strict=True)
        pretrain_model = transfer_multihead_model(p,pretrain_model,pretrained['head'])
    elif p['pretrain_type'] == 'spice':
        pretrain_model = load_spice_model(pretrain_model,p['pretrain_path'],p['model_type'])        
        pretrain_model = transfer_multihead_model(p,pretrain_model)
    else: raise ValueError('not implemented')
    """

    backbone = get_backbone(p,secondary=True) # OK
    model = get_head_model(p,backbone,secondary=True) # OK

    if p['fixmatch_model']['pretrain_type'] == 'scan':
        pretrained = torch.load(p['fixmatch_model']['pretrain_path'],map_location='cpu')
        model.load_state_dict(pretrained['model'],strict=True)
        model = transfer_multihead_model(p,model,pretrained['head'])
        model_type = 'cluster_head'
    elif p['fixmatch_model']['pretrain_type'] == 'spice':
        model = load_spice_model(model,p['fixmatch_model']['pretrain_path'],p['fixmatch_model']['model_type'])        
        model = transfer_multihead_model(p,model)
        model_type = 'cluster_head'
    else:
        model_dict = torch.load(p['fixmatch_model']['pretrain_path'],map_location='cpu')
        model.load_state_dict(model_dict,strict=True)
        model_type = 'fixmatch_model'

    optimizer = get_optimizer(p,model)

    #labeled_data.evaluate_samples(p,pretrain_model)

    label_step = len(labeled_data)//p['batch_size']
    unlabeled_step = len(unlabeled_data)//p['batch_size']
    step_size = min(label_step,unlabeled_step)

    labeled_loader = torch.utils.data.DataLoader(labeled_data, num_workers=p['num_workers'], 
                                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                drop_last=True, shuffle=True)

    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_data, num_workers=p['num_workers'], 
                                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                drop_last=True, shuffle=True)

    return {'base_dataloader': base_dataloader, 'label_dataloader': labeled_loader, 'unlabeled_dataloader': unlabeled_loader, 'validation_loader': val_loader['val_loader'], 'step_size': step_size, 'optimizer': optimizer, 'model': model, 'model_type': model_type}
    


#@author: Michael Blachetta
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

