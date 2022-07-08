import argparse
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
from evaluate import evaluate_singleHead, Analysator
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
import pandas as pd

from functionality import initialize_training


FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier')
FLAGS.add_argument('-config',help='path to the trial list')
FLAGS.add_argument('-p',help='prefix')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')
FLAGS.add_argument('--model_path', help='path to the model files')
FLAGS.add_argument('-loss_track',help='wether to track loss value',default='no')

def main():
    args = FLAGS.parse_args()
    p = create_config(args.p ,args.root_dir, args.config, args.p)
    prefix = args.p
    p['prefix'] = args.p
    gpu_id = args.gpu
    num_cluster = p['num_classes']
    fea_dim = p['feature_dim']
    p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
    backbone_file = p['pretrain_path']
    p['pretrain_path'] = os.path.join(args.model_path,backbone_file)
    params = initialize_training(p)
    last_loss = 1000

    if args.loss_track in ['y','yes']:
        start_info, end_info, new_optimum = loss_track_session(prefix,params,p,prefix,last_loss,gpu_id)
    else:
        start_info, end_info, new_optimum = general_session(prefix,params,p,prefix,last_loss,gpu_id)

    session_stats = statisics_register()
    session_stats.add_session_statistic(0,p,start_info,end_info)
    torch.save(session_stats,'EVALUATION/'+prefix+'/'+prefix+'.session')




def loss_track_session(rID,components,p,prefix,last_loss,gpu_id=0):

    end_epoch = p['epochs']

    batch_loader = components['train_dataloader']
    model = components['model']
    first_criterion = components['criterion']
    optimizer = components['optimizer']        
    second_criterion = components['second_criterion']
    train_one_epoch = components['train_method']
    val_loader = components['val_dataloader']


    train_method = p['train_method']
    p['train_args'] = {}
    p['train_args']['device'] = 'cuda'
    p['train_args']['gpu_id'] = gpu_id
    p['train_args']['update_cluster_head_only'] = p['update_cluster_head_only']
    p['train_args']['local_crops_number'] = p['augmentation_kwargs']['local_crops_number']
    p['train_args']['aug'] = p['augmentation_strategy']


    start_epoch = 0
    best_loss = 1e4
    best_loss_head = 0
    best_epoch = 0
    best_accuracy = 0

    device_id = 'cuda:'+str(gpu_id)


    if p['setup'] == 'scan':
        
        loss_track = pd.DataFrame({

            'epoch': [],
            'entropy_loss': [],
            'consistency_loss': [],
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

    # first epoch

    c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=0, train_args=p['train_args'], second_criterion=second_criterion)
    
    if p['num_heads'] > 1:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
        starting_data.compute_kNN_statistics(100)
        starting_data.compute_real_consistency(0.5)
        start_stats = starting_data.return_statistic_summary(c_loss)
    else:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,val_loader)
        starting_data.compute_kNN_statistics(100)
        starting_data.compute_real_consistency(0.5)
        start_stats = starting_data.return_statistic_summary(c_loss)   

    # Main loop

    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch+1, end_epoch):
        print(colored('Epoch %d/%d' %(epoch+1, end_epoch), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=epoch, train_args=p['train_args'], second_criterion=second_criterion)

        # evaluate
        pdict = {'device_id': device_id, 'val_loader': val_loader, 'model': model, 'loss_track': loss_track, 'best_loss': best_loss, 'best_head': best_head, 'c_loss': c_loss, 'best_loss_head': best_loss_head, 'best_epoch': best_epoch,  'epoch': epoch, 'prefix': prefix  }
        parameter = evaluate_loss_track(p,pdict)
        best_loss  = parameter['best_loss']
        best_loss_head = parameter['best_loss_head']
        best_epoch     = parameter['best_epoch']
        loss_track     = parameter['loss_track']
       
 
    print('-------------------SESSION COMPLETED--------------------------')

    loss_track.to_csv('EVALUATION/'+rID+'/'+prefix+'_loss_statistics.csv')
    print('best_epoch: ',best_epoch)

    if p['num_heads'] > 1: 
        if train_method == 'scan':
            # Evaluate and save the final model
            print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
            model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
            model.load_state_dict(model_checkpoint['model']) # Ab hier ist model optimal

            metric_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
            metric_data.compute_kNN_statistics(100)
            metric_data.compute_real_consistency(0.5)
            session_stats = metric_data.return_statistic_summary(best_loss)
            predictions = get_predictions(device_id, p, val_loader, model)
            clustering_stats = hungarian_evaluate(device_id, model_checkpoint['head'], predictions,
                                    class_names=val_loader.dataset.classes,
                                    compute_confusion_matrix=True,
                                    confusion_matrix_file=os.path.join(p['scan_dir'],prefix+'_confusion_matrix.png'))

            torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')

            if best_loss > last_loss:
                return start_stats ,session_stats, True
            else:
                return start_stats ,session_stats, False

        else:
            print('best accuracy: ',best_accuracy)
            print('head_id: ',best_loss_head)
            print('\n')
            best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
            model.load_state_dict(best_copy)
            metric_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
            metric_data.compute_kNN_statistics(100)
            metric_data.compute_real_consistency(0.5)
            session_stats = metric_data.return_statistic_summary(best_loss)
            torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')

            if best_loss > last_loss:
                return start_stats ,session_stats, True
            else:
                return start_stats ,session_stats, False

    else: 

        best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
        model.load_state_dict(best_copy)
        metric_data = Analysator(device_id,model,val_loader)
        metric_data.compute_kNN_statistics(100)
        metric_data.compute_real_consistency(0.5)
        session_stats = metric_data.return_statistic_summary(best_loss)
        torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')

        if best_loss > last_loss:
            return start_stats ,session_stats, True
        else:
            return start_stats ,session_stats, False


def general_session(rID,components,p,prefix,last_loss,gpu_id=0): #------------------------------------------------------

    end_epoch = p['epochs']
    #prefix = p['prefix']
    batch_loader = components['train_dataloader']
    model = components['model']
    first_criterion = components['criterion']
    optimizer = components['optimizer']        
    second_criterion = components['second_criterion']
    train_one_epoch = components['train_method']
    val_loader = components['val_dataloader']


    train_method = p['train_method']
    p['train_args'] = {}
    p['train_args']['device'] = 'cuda'
    p['train_args']['gpu_id'] = gpu_id
    p['train_args']['update_cluster_head_only'] = p['update_cluster_head_only']
    p['train_args']['local_crops_number'] = p['augmentation_kwargs']['local_crops_number']
    p['train_args']['aug'] = p['augmentation_strategy']

    start_epoch = 0
    best_loss = 1e4
    best_loss_head = 0
    best_epoch = 0
    best_accuracy = 0
    device_id = 'cuda:'+str(gpu_id)


    # start_epoch, best_loss, best_loss_head = resume_from_checkpoint(model,optimizer,p['scan_checkpoint'])

    c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=0, train_args=p['train_args'], second_criterion=second_criterion)
    if p['num_heads'] > 1:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
        starting_data.compute_kNN_statistics(100)
        starting_data.compute_real_consistency(0.5)
        start_stats = starting_data.return_statistic_summary(c_loss)
    else:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,val_loader)
        starting_data.compute_kNN_statistics(100)
        starting_data.compute_real_consistency(0.5)
        start_stats = starting_data.return_statistic_summary(c_loss)


    # Main loop
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch+1, end_epoch):
        print(colored('Epoch %d/%d' %(epoch+1, end_epoch), 'yellow'))
        print(colored('-'*15, 'yellow'))
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        # Train
        print('Train ...')
        c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=epoch, train_args=p['train_args'], second_criterion=second_criterion)
        # evaluate
        pdict = {'model': model, 'best_head': best_head, 'c_loss': c_loss, 'best_loss': best_loss,'best_loss_head': best_loss_head, 'best_epoch': best_epoch, 'epoch': epoch, 'prefix': prefix }
        parameter = evaluate_standard(p,pdict)
        best_loss  = parameter['best_loss']
        best_loss_head = parameter['best_loss_head']
        best_epoch     = parameter['best_epoch']
       
 
    print('-------------------SESSION COMPLETED--------------------------')
    print('best_epoch: ',best_epoch)

    if p['num_heads'] > 1: 
        if train_method == 'scan':
            # Evaluate and save the final model
            print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
            model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
            model.load_state_dict(model_checkpoint['model']) # Ab hier ist model optimal
            metric_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
            metric_data.compute_kNN_statistics(100)
            metric_data.compute_real_consistency(0.5)
            run_statistics = metric_data.return_statistic_summary(best_loss)
            #recent_entropy = run_statistics['entropy']  difficult to find model with best loss and entropy

            if best_loss > last_loss:
                torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')
                print('next_loss: ',best_loss,';  ',prefix)
                return start_stats,run_statistics, True
            
             # metric_data.return_statistic_summary(best_loss)
            return start_stats, run_statistics, False
             
            

        else:
            print('best accuracy: ',best_accuracy)
            print('head_id: ',best_loss_head)
            print('\n')
            best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
            model.load_state_dict(best_copy)
            metric_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
            metric_data.compute_kNN_statistics(100)
            metric_data.compute_real_consistency(0.5)

            if best_loss > last_loss:
                torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')
                print('next_loss: ',best_loss,';  ',prefix)
                return start_stats ,metric_data.return_statistic_summary(best_loss), True

            return start_stats ,metric_data.return_statistic_summary(best_loss), False

    else: 

        best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
        model.load_state_dict(best_copy)
        metric_data = Analysator(device_id,model,val_loader)
        metric_data.compute_kNN_statistics(100)
        metric_data.compute_real_consistency(0.5)

        if best_loss > last_loss:
            torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')
            print('next_loss: ',best_loss,';  ',prefix)
            return start_stats ,metric_data.return_statistic_summary(best_loss), True

        return start_stats ,metric_data.return_statistic_summary(best_loss), False


def evaluate_loss_track(p,parameters):

    train_method = p['train_method']
    loss_track = parameters['loss_track']
    val_loader = parameters['val_loader']
    model      = parameters['model']
    device_id  = parameters['device_id']
    best_loss  = parameters['best_loss']
    best_head  = parameters['best_head']
    c_loss     = parameters['c_loss']
    best_loss_head = parameters['best_loss_head']
    best_epoch     = parameters['best_epoch']
    epoch = parameters['epoch']
    prefix = parameters['prefix']

    if p['num_heads'] > 1: 
        if train_method == 'scan':
            print('Make prediction on validation set ...')
            predictions = get_predictions(device_id, p, val_loader, model)
            print('Evaluate based on SCAN loss ...')
            scan_stats = scan_evaluate(predictions)
            print(scan_stats)
            epoch_stats = {}
            loss_metrics = scan_stats['scan'][best_loss_head]
            detailed_metrics = evaluate_singleHead(device_id,model,val_loader,forwarding='singleHead_eval')
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
            detailed_metrics = evaluate_singleHead(device_id,model,val_loader,forwarding='singleHead_eval')
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
                torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')
                print('\nLOSS accuracy: ', best_accuracy,'  on head ',best_loss_head)
            

             #if p['update_cluster_head_only']:
            #    torch.save( model.cluster_head[best_loss_head].state_dict(), 'PRODUCTS/'+prefix+'_best_mlpHead.pth')
            #else:
            
            
        else:
            print('No new lowest loss on validation set')
            print('Lowest loss head is %d' %(best_loss_head))

    else: 
        detailed_metrics = evaluate_singleHead(device_id,model,val_loader)
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
            #if p['update_cluster_head_only']:
            #    torch.save(model.head.state_dict(),'PRODUCTS/'+prefix+'_best_mlpHead.pth')                
            #else:
            torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')
 

        # Checkpoint
    #print('Checkpoint ...')
    #torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head}, p['scan_checkpoint'])
    pdict = {}
    pdict['loss_track'] = loss_track
    pdict['best_loss'] = best_loss
    pdict['best_head'] = best_head 
    pdict['c_loss'] = c_loss
    pdict['best_loss_head'] = best_loss_head
    pdict['best_epoch'] = best_epoch
    
    return pdict 
  
    
def evaluate_standard(p,parameters):

    model = parameters['model']
    best_loss = parameters['best_loss']
    best_head = parameters['best_head']
    c_loss = parameters['c_loss']
    best_loss_head = parameters['best_loss_head']
    best_epoch = parameters['best_epoch']
    epoch = parameters['epoch']
    prefix = parameters['prefix']
    

    train_method = p['train_method']

    if p['num_heads'] > 1:          
        
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
                torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')
                #print('\nLOSS accuracy: ', best_accuracy,'  on head ',best_loss_head)
           
            #if p['update_cluster_head_only']:
            #    torch.save( model.cluster_head[best_loss_head].state_dict(), 'PRODUCTS/'+prefix+'_best_mlpHead.pth')
            #else:
            
            
        else:
            print('No new lowest loss on validation set')
            print('Lowest loss head is %d' %(best_loss_head))

    else: 

        if c_loss < best_loss:
            best_loss = c_loss
            #if p['update_cluster_head_only']:
            #    torch.save(model.head.state_dict(),'PRODUCTS/'+prefix+'_best_mlpHead.pth')                
            #else:
            torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')
            
    pdict = {}
    pdict['best_loss'] = best_loss
    pdict['best_head'] = best_head
    pdict['c_loss'] = c_loss
    pdict['best_loss_head'] = best_loss_head
    pdict['best_epoch'] = best_epoch
    
    return pdict


def compute_runlist(combinations,prefixes,init_dict,keylist,key_names,session_list):
        
    if keylist:
        k = keylist.pop()
        for v in combinations[k]:
            init_dict[k] = v
            if k in prefixes.keys():
                key_names[k] = '_'+prefixes[k]+str(v)
            compute_runlist(combinations,prefixes,init_dict,copy.deepcopy(keylist),key_names,session_list)               
        
    else:
        parameters = copy.deepcopy(init_dict)
        parameters['name'] = prefixes['base_name']
        for k in key_names.keys():
            parameters['name'] += key_names[k]
        session_list.append(parameters)


def store_statistic_analysis(p,model,val_loader,prefix,best_loss): # needs to create folder ANALYSIS
    
    gpu_id = p['train_args']['gpu_id']
    device_id = 'cuda:'+str(gpu_id)

    if p['num_heads'] > 1:
        data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
        data.compute_kNN_statistics(100)
        data.compute_real_consistency(0.5)
        data.return_statistic_summary(best_loss)
    else:
        data = Analysator(device_id,model,val_loader)
        data.compute_kNN_statistics(100)
        data.compute_real_consistency(0.5)
        data.return_statistic_summary(best_loss)

    torch.save({'analysator': data ,'parameter':p},'ANALYSIS/'+prefix+'_ANALYSE')

#def load_trial_list(trial_file):


class statisics_register():

    def __init__(self):
        self.meta_track = []
        self.start_track = []
        self.performance_track = []
        self.difference_table = pd.DataFrame()
        self.hyperparameter_table = pd.DataFrame()
        self.output_table = pd.DataFrame()
        self.session_table = pd.DataFrame()

    def add_session_statistic(self,id,para,start_series,end_series):
     
        hyperparams = pd.Series()
        results = pd.Series()
        delta = pd.Series()

        idx = para['prefix']+'_'+str(id)

        self.performance_track.append(end_series)
        self.start_track.append(start_series)
        self.output_table = pd.concat([self.output_table,pd.DataFrame(dict(end_series),columns=end_series.index,index=[idx])])

        

        hyperparams['augmentation_strategy'] = para['augmentation_strategy']
        hyperparams['backbone'] = para['backbone']
        hyperparams['head_type'] = para['model_args']['head_type']
        hyperparams['batch_norm'] = para['model_args']['batch_norm']
        hyperparams['optimizer'] = para['optimizer']
        hyperparams['full_model'] = para['update_cluster_head_only'] 
        hyperparams['epochs'] = para['epochs']
        hyperparams['batch_size'] = para['batch_size']
        hyperparams['feature_dim'] = para['feature_dim']
        hyperparams['num_heads'] = para['num_heads']
        
        self.meta_track.append(hyperparams)
        info_part = pd.DataFrame(dict(hyperparams),columns=hyperparams.index,index=[idx])
        self.hyperparameter_table = pd.concat([self.hyperparameter_table, info_part])

        for key in start_series.index:

            s = 'start_'+str(key)
            e = 'end_'+str(key)
            d = 'd_'+str(key)

            results[s] = start_series[key]
            results[e] = end_series[key]
            delta[d] = end_series[key] - start_series[key]
            results[d] = end_series[key] - start_series[key]

        self.difference_table = pd.concat([self.difference_table,pd.DataFrame(dict(delta),columns=delta.index,index=[idx])])

        data_part = pd.DataFrame(dict(results),columns=results.index,index=[idx])

        next_row = pd.concat({'settings':info_part,'results': data_part},axis=1,names=['part:','values:'])
            
        return pd.concat([self.session_table,next_row])
