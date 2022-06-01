import argparse
import os
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
from utils.config import create_config
from utils.common_config import adjust_learning_rate
from models import load_backbone_model
from evaluate import evaluate_singleHead, evaluate_headlist
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
import pandas as pd


FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier')
FLAGS.add_argument('-p',help='prefix file selection')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--pretrain_path', help='filename or path of the pretrained model')


args = FLAGS.parse_args()
p = create_config(args.root_dir, args.config_exp, args.p)
prefix = args.p
gpu_id = args.gpu

num_cluster = p['num_classes']
fea_dim = p['feature_dim']
#p['last_activation']

p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
#args_head = {'num_neurons': [fea_dim, fea_dim, num_cluster], 'last_activation': p['last_activation'], 'aug_type':p['aug_type'], 'last_batchnorm': p['last_batchnorm']}

#args = FLAGS.parse_args()

 # CUDNN
torch.backends.cudnn.benchmark = True

    # Data
#print(colored('Get dataset and dataloaders', 'blue'))

"""------------- Augmentation & Transformation --------------------"""

augmentation_method = p['augmentation_strategy']
augmentation_type = p['augmentation_type']

if augmentation_type == 'scan':

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
    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))

else:
    aug_args = p['augmentation_kwargs']
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

#train_transformations = get_train_transformations(p)

val_transformations = transforms.Compose([
                                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(**p['transformation_kwargs']['normalize'])])


""" --------------- TRAINING: Select Dataset -------------------- """
# dataset:
from dataset.scan_dataset import STL10_trainNtest
#
split = p['split']
dataset_type = p['dataset_type']

if dataset_type == 'scan':

    if p['train_db_name'] == 'cifar-10':
        from dataset.scan_dataset import CIFAR10
        dataset = CIFAR10(train=True, transform=train_transformation, download=True)
        eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

    elif p['train_db_name'] == 'cifar-20':
        from dataset.scan_dataset import CIFAR20
        dataset = CIFAR20(train=True, transform=train_transformation, download=True)
        eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)

    elif p['train_db_name'] == 'stl-10':
        from dataset.scan_dataset import STL10
        dataset = STL10(split=split, transform=train_transformation, download=False)
        #eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)
        eval_dataset = STL10(split='train',transform=val_transformations,download=False)

    elif p['train_db_name'] == 'imagenet':
        from dataset.scan_dataset import ImageNet
        dataset = ImageNet(split='train', transform=train_transformation)

    elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from dataset.scan_dataset import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=val_transformations)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if p['to_augmented_dataset']: # Dataset returns an image and an augmentation of that image.
        from dataset.scan_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if p['to_neighbors_dataset']: # Dataset returns an image and one of its nearest neighbors.
        from dataset.scan_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])
        #indices_ = np.load(p['topk_neighbors_train_path'])
        eval_dataset = NeighborsDataset(eval_dataset, indices, p['num_neighbors'])

else:
    dataset = torchvision.datasets.STL10('/space/blachetta/data', split=split,transform=train_transformation, download=True)
    eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)

### data_loader:

""" Custom collate function """
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


if dataset_type == 'scan':

    batch_loader = torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)

else:
    batch_loader = torch.utils.data.DataLoader(dataset,num_workers=p['num_workers'],batch_size=p['batch_size'],pin_memory=True,drop_last=True,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True,
            drop_last=False, shuffle=False)

""" ------------- Build MODEL ------------------"""

class wrapped_resnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self,x):
        return self.backbone(x).flatten(start_dim=1)

backbone_model_ID = p['backbone']

# get Backbone model

if backbone_model_ID == 'lightly_resnet18':
    resnet = torchvision.models.resnet18()
    coreModel = nn.Sequential(*list(resnet.children())[:-1])
    backbone = wrapped_resnet(coreModel)
    backbone_outdim = 512

elif backbone_model_ID == 'clPcl':
    from models import resnet18
    backbone = resnet18()
    backbone_outdim = 512

elif backbone_model_ID == 'scatnet':
    from scatnet import ScatSimCLR
    backbone = ScatSimCLR(J=p['scatnet_args']['J'], L=p['scatnet_args']['L'], input_size=tuple(p['scatnet_args']['input_size']), res_blocks=p['scatnet_args']['res_blocks'],out_dim=p['scatnet_args']['out_dim'])
    backbone_outdim = p['scatnet_args']['out_dim']

else: raise ValueError

load_backbone_model(backbone,args.pretrain_path,backbone_model_ID)

model_type = p['model_type']
p['model_args']['nheads'] = p['num_heads']
model_args = p['model_args']
#model_args
hidden_dim = p['hidden_dim']
num_cluster = p['num_classes']

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

else: raise ValueError


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

elif loss_ID == 'selflabel':
    from loss import ConfidenceBasedCE
    first_criterion = ConfidenceBasedCE(p['loss_args']['threshold'], p['loss_args']['apply_class_balancing'])

else: raise ValueError



""" --------------- TRAINING --------------- """

train_method = p['train_method']
p['train_args'] = {}
p['train_args']['device'] = 'cuda'
p['train_args']['gpu_id'] = gpu_id
p['train_args']['update_cluster_head_only'] = p['update_cluster_head_only']
p['train_args']['local_crops_number'] = p['augmentation_kwargs']['local_crops_number']
p['train_args']['aug'] = p['augmentation_strategy']

train_one_epoch = None

if train_method == 'scan':
    from training import scan_train
    train_one_epoch = scan_train
    
elif train_method == 'twist':
    from training import twist_training
    train_one_epoch = twist_training

elif train_method == 'simclr':
    from training import simclr_train
    train_one_epoch = simclr_train

elif train_method == 'selflabel':
    from training import selflabel_train
    train_one_epoch = selflabel_train

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

start_epoch = 0
best_loss = 1e4
best_loss_head = 0
best_epoch = 0
best_accuracy = 0
head_accuracy = None
best_model_dict = {}
device_id = 'cuda:'+str(gpu_id)

print('FIRST CHECKPOINT PASSED')

if p['setup'] == 'scan':
    
    loss_track = pd.DataFrame({

        'epoch': [],
        'entropy_loss': [],
        'consostency_loss': [],
        'total_loss': [],
        'Accuracy': [],
        'Adjusted_Mutual_Information': [],
        'Adjusted_Random_Index': [],
        'V_measure': [],
        'fowlkes_mallows': [],
        'cluster_size_entropy': [],
        'consistency_ratio': [],
        'confidence_ratio': [],
        'mean_confidence': [], 
        'std_confidence': []
    
    })

else:

    loss_track = pd.DataFrame({

        'epoch': [],
        'total_loss': [],
        'Accuracy': [],
        'Adjusted_Mutual_Information': [],
        'Adjusted_Random_Index': [],
        'V_measure': [],
        'fowlkes_mallows': [],
        'cluster_size_entropy': [],
        'consistency_ratio': [],
        'confidence_ratio': [],
        'mean_confidence': [], 
        'std_confidence': []
    
    })

# start_epoch, best_loss, best_loss_head = resume_from_checkpoint(model,optimizer,p['scan_checkpoint'])

    # Main loop
print(colored('Starting main loop', 'blue'))

for epoch in range(start_epoch, p['epochs']):
    print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
    print(colored('-'*15, 'yellow'))

        # Adjust lr
    lr = adjust_learning_rate(p, optimizer, epoch)
    print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
    print('Train ...')
    c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=epoch, train_args=p['train_args'], second_criterion=second_criterion)


    # evaluate

    if p['num_heads'] > 1: 
        if train_method == 'scan':
            print('Make prediction on validation set ...')
            predictions = get_predictions(device_id, p, val_dataloader, model)
            print('Evaluate based on SCAN loss ...')
            scan_stats = scan_evaluate(predictions)
            print(scan_stats)
            epoch_stats = {}
            loss_metrics = scan_stats['scan'][best_loss_head]
            detailed_metrics = evaluate_singleHead(device_id,model,val_dataloader,forwarding='singleHead_eval')
            epoch_stats['epoch'] = epoch
            epoch_stats['entropy_loss'] = loss_metrics['entropy']
            epoch_stats['consistency_loss'] = loss_metrics['consistency']
            epoch_stats['total_loss'] = loss_metrics['total_loss']
            epoch_stats['Accuracy'] = detailed_metrics['Accuracy']
            epoch_stats['Adjusted_Mutual_Information'] = detailed_metrics['Adjusted_Mutual_Information']
            epoch_stats['Adjusted_Random_Index'] = detailed_metrics['Adjusted_Random_Index']
            epoch_stats['V_measure'] = detailed_metrics['V_measure']
            epoch_stats['fowlkes_mallows'] = detailed_metrics['fowlkes_mallows']
            epoch_stats['cluster_size_entropy'] = detailed_metrics['cluster_size_entropy']
            epoch_stats['confidence_ratio'] = detailed_metrics['confidence_ratio']
            epoch_stats['mean_confidence'] = detailed_metrics['mean_confidence']
            epoch_stats['std_confidence'] = detailed_metrics['std_confidence']
            epoch_stats['consistency_ratio'] = detailed_metrics['consistency_ratio']
            loss_track = loss_track.append(epoch_stats,ignore_index=True)
            
            #lowest_loss_head = scan_stats['lowest_loss_head']
            #lowest_loss = scan_stats['lowest_loss']
        else: #result_dicts = evaluate_headlist(device_id,model,val_dataloader)
            detailed_metrics = evaluate_singleHead(device_id,model,val_dataloader,forwarding='singleHead_eval')
            epoch_stats = {}
            epoch_stats['epoch'] = epoch
            epoch_stats['total_loss'] = c_loss
            epoch_stats['Accuracy'] = detailed_metrics['Accuracy']
            epoch_stats['Adjusted_Mutual_Information'] = detailed_metrics['Adjusted_Mutual_Information']
            epoch_stats['Adjusted_Random_Index'] = detailed_metrics['Adjusted_Random_Index']
            epoch_stats['V_measure'] = detailed_metrics['V_measure']
            epoch_stats['fowlkes_mallows'] = detailed_metrics['fowlkes_mallows']
            epoch_stats['cluster_size_entropy'] = detailed_metrics['cluster_size_entropy']
            epoch_stats['confidence_ratio'] = detailed_metrics['confidence_ratio']
            epoch_stats['mean_confidence'] = detailed_metrics['mean_confidence']
            epoch_stats['std_confidence'] = detailed_metrics['std_confidence']
            epoch_stats['consistency_ratio'] = detailed_metrics['consistency_ratio']
            loss_track = loss_track.append(epoch_stats,ignore_index=True)           

        
        if c_loss < best_loss:
            print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, c_loss))
            print('Lowest loss head is %d' %(best_head))
            best_loss = c_loss
            best_loss_head = best_head.item()
            model.best_head_id = best_loss_head
            best_epoch = epoch

            if train_method == 'scan':
                torch.save({'model': model.state_dict(), 'head': best_loss_head}, p['scan_model'])
            else:
                #result = result_dicts[best_loss_head]
                #best_accuracy = result['ACC']
                print('\nLOSS accuracy: ', best_accuracy,'  on head ',best_loss_head)
                #best_model_dict = result

            if p['update_cluster_head_only']:
                torch.save( model.cluster_head[best_loss_head].state_dict(), prefix+'_best_mlpHead.pth')
            else:
                torch.save(model.state_dict(),prefix+'_best_MODEL')
            
        else:
            print('No new lowest loss on validation set')
            print('Lowest loss head is %d' %(best_loss_head))

    else: 
        detailed_metrics = evaluate_singleHead(device_id,model,val_dataloader)
        epoch_stats['epoch'] = epoch
        epoch_stats['total_loss'] = c_loss
        epoch_stats['Accuracy'] = detailed_metrics['Accuracy']
        epoch_stats['Adjusted_Mutual_Information'] = detailed_metrics['Adjusted_Mutual_Information']
        epoch_stats['Adjusted_Random_Index'] = detailed_metrics['Adjusted_Random_Index']
        epoch_stats['V_measure'] = detailed_metrics['V_measure']
        epoch_stats['fowlkes_mallows'] = detailed_metrics['fowlkes_mallows']
        epoch_stats['cluster_size_entropy'] = detailed_metrics['cluster_size_entropy']
        epoch_stats['confidence_ratio'] = detailed_metrics['confidence_ratio']
        epoch_stats['mean_confidence'] = detailed_metrics['mean_confidence']
        epoch_stats['std_confidence'] = detailed_metrics['std_confidence']
        loss_track = loss_track.append(epoch_stats,ignore_index=True) 

        if detailed_metrics['Accuracy'] > best_accuracy:
            best_accuracy = detailed_metrics['Accuracy']
            best_epoch = epoch
            print('new top accuracy: ',best_accuracy)

        if c_loss < best_loss:
            best_loss = c_loss

            if p['update_cluster_head_only']:
                torch.save(model.head.state_dict(),prefix+'_best_mlpHead.pth')                
            else:
                torch.save(model.state_dict(),prefix+'_best_model.pth')

            

    #print('Evaluate with hungarian matching algorithm ...')
    #clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
    #print(clustering_stats)     

        # Checkpoint
    print('Checkpoint ...')
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head}, p['scan_checkpoint'])


loss_track.to_csv(prefix+'_loss_statistics.csv')
print('best_epoch: ',best_epoch)

if p['num_heads'] > 1: 
    if train_method == 'scan':
        # Evaluate and save the final model
        print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
        model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
        model.load_state_dict(model_checkpoint['model'])
        predictions = get_predictions(device_id, p, val_dataloader, model)
        clustering_stats = hungarian_evaluate(device_id, model_checkpoint['head'], predictions,
                                class_names=eval_dataset.dataset.classes,
                                compute_confusion_matrix=True,
                                confusion_matrix_file=os.path.join(p['scan_dir'],prefix+'_confusion_matrix.png'))

        print(clustering_stats)

    else:
        print('best accuracy: ',best_accuracy)
        print('head_id: ',best_loss_head)
        print('\n')
        #result = evaluate_headlist(device_id,model,val_dataloader)
        for k in best_model_dict.keys():
            print(k,': ',best_model_dict[k])
        #print(best_model_dict)

else: 

    result = evaluate_singleHead(device_id,model,val_dataloader)
    print('best_loss: ',best_loss)
    print('best_accuracy: ',best_accuracy)  
    print('FINAL MODEL:')
    print('ACC: ',result['ACC'])
    print('ARI: ',result['ARI'])
    print('AMI: ',result['AMI'])