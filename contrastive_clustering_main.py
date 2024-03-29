import argparse
from torch._six import string_classes
int_classes = int
from utils.config import create_config
from evaluate import logger
import pandas as pd
from functionality import initialize_contrastive_clustering
import os
import copy
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
from evaluate import evaluate_singleHead, Analysator, logger
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from execution_utils import evaluation
import pandas as pd
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier')
FLAGS.add_argument('-config',help='path to the config file')
FLAGS.add_argument('-p',help='prefix')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')

args = FLAGS.parse_args()

mkdir_if_missing('EVALUATION/'+args.p)
with open(args.config, 'r') as stream:
    config = yaml.safe_load(stream)
    p = EasyDict()
    # Copy
    for k, v in config.items():
        p[k] = v


prefix = args.p
p['prefix'] = args.p
gpu_id = args.gpu
p['device'] = 'cuda:'+str(gpu_id)
p['hidden_dim'] = 128
p['num_heads'] = 0
p['model_args'] = {'nheads': 0}
end_epoch = p['epochs']
p['update_cluster_head_only'] = False
p['rID'] = prefix

components = initialize_contrastive_clustering(p)




unlabeled_loader = components['unlabeled_dataloader']
labeled_loader = components['labeled_dataloader']
instance_criterion = components['instance_criterion']
cluster_criterion = components['cluster_criterion']
model = components['model']
optimizer = components['optimizer']        
train_one_epoch = components['train_method']
val_loader = components['val_dataloader']

best_loss = 10000


 # Main loop
print(colored('Starting main loop', 'blue'))
for epoch in range(0, end_epoch):
    print(colored('Epoch %d/%d' %(epoch, end_epoch), 'yellow'))
    print(colored('-'*15, 'yellow'))
    # Adjust lr
    lr = adjust_learning_rate(p, optimizer, epoch)
    print('Adjusted learning rate to {:.5f}'.format(lr))
    # Train
    print('Train ...')
    c_loss, _ = train_one_epoch(p['device'], model, optimizer, unlabeled_loader, labeled_loader, instance_criterion, cluster_criterion)


    print('loss = ', c_loss)
    if c_loss < best_loss:
        print('new best loss: ',best_loss,' --> ',c_loss)
        best_loss = c_loss
        torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')


_, results, _ = evaluation(p['device'],p,model,val_loader,None,best_loss,0,model_type='contrastive_clustering')

print(results)

    