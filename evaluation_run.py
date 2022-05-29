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
FLAGS.add_argument('-p',help='prefix file selection')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--model_path', help='path to the model files')


def main():

    args = FLAGS.parse_args()
    p = create_config(args.root_dir, args.config_exp, args.p)
    prefix = args.p
    gpu_id = args.gpu
    num_cluster = p['num_classes']
    fea_dim = p['feature_dim']
    p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
    backbone_file = p['pretrain_path']
    p['pretrain_path'] = os.path.join(args.model_path,backbone_file)
    params = initialize_training(p)

    start_info, end_info, new_optimum = loss_track_session(params,p,prefix,gpu_id)

    last_loss = 1000

    from base_config import session_list_scan
    i = 0
    run_id = prefix
    for session in session_list_scan:
        run_id = prefix+str(i)
        p = map_parameters(p,session,args.model_path)
        params = initialize_training(p)
        if session['save_data']:
            start_info, end_info, new_optimum = loss_track_session(params,p,run_id,gpu_id)
            if new_optimum: last_loss = end_info['loss']
        else:
            start_info, end_info, new_optimum = general_session(params,p,run_id,last_loss,gpu_id)
            if new_optimum: 
                last_loss = end_info['loss']
            else: # delete files
                if p['train_method'] == 'scan':
                    if os.path.exists(p['scan_model']): os.remove(p['scan_model'])
                else:
                    if os.path.exists('PRODUCTS/'+run_id+'_best_model.pth'): os.remove('PRODUCTS/'+run_id+'_best_model.pth')
        i += 1

    



#load_backbone_model(backbone,os.path.join(args.pretrain_path,p['backbone_file']),backbone_model_ID)



def loss_track_session(components,p,prefix,gpu_id=0):

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

    c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=epoch, train_args=p['train_args'], second_criterion=second_criterion)
    
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

    loss_track.to_csv('EVALUATION/'+prefix+'_loss_statistics.csv')
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

            torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+prefix+'_ANALYSATOR')

            return start_stats ,session_stats

        else:
            print('best accuracy: ',best_accuracy)
            print('head_id: ',best_loss_head)
            print('\n')
            best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
            model.load_state_dict(best_copy)
            metric_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval')
            metric_data.compute_kNN_statistics(100)
            metric_data.compute_real_consistency(0.5)
            torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+prefix+'_ANALYSATOR')

            return start_stats ,metric_data.return_statistic_summary(best_loss)

    else: 

        best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
        model.load_state_dict(best_copy)
        metric_data = Analysator(device_id,model,val_loader)
        metric_data.compute_kNN_statistics(100)
        metric_data.compute_real_consistency(0.5)
        torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+prefix+'_ANALYSATOR')

        return start_stats, metric_data.return_statistic_summary(best_loss)



def general_session(components,p,prefix,last_loss,gpu_id=0): #------------------------------------------------------

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

    c_loss, best_head = train_one_epoch(train_loader=batch_loader, model=model, criterion=first_criterion, optimizer=optimizer, epoch=epoch, train_args=p['train_args'], second_criterion=second_criterion)
    
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
        pdict = {'model': model, 'best_head': best_head, 'c_loss': c_loss, 'best_loss_head': best_loss_head, 'best_epoch': best_epoch, 'epoch': epoch, 'prefix': prefix }
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
                torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+prefix+'_ANALYSATOR')
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
                torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+prefix+'_ANALYSATOR')
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
            torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+prefix+'_ANALYSATOR')
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


def map_parameters(p,params,model_path):
    
    for attribute in params.keys():
        if '.' in attribute:
            primary, secondary = attribute.split('.')
            p[primary][secondary] = params[attribute]
        else:
            p[attribute] = params[attribute]
        
        num_cluster = p['num_classes']
        fea_dim = p['feature_dim']
        p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
        backbone_file = p['pretrain_path']
        p['pretrain_path'] = os.path.join(model_path,backbone_file)

        return p
        
        


if __name__ == "__main__":
    main()