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
from evaluate import evaluate_singleHead, Analysator, logger
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
import pandas as pd


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
    p['rID'] = rID
    p['prefix'] = prefix
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
    
    if p['train_split'] in ['train','test']:
        Vdataloader = val_loader['train_split'] if p['train_split'] == 'test' else val_loader['test_split']
    else: Vdataloader = val_loader['val_loader']

    if p['num_heads'] > 1:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,Vdataloader,forwarding='singleHead_eval')
        starting_data.compute_kNN_statistics(100)
        starting_data.compute_real_consistency(0.5)
        start_stats = starting_data.return_statistic_summary(c_loss)
    else:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,Vdataloader)
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
        pdict = {'device_id': device_id, 'val_loader': Vdataloader, 'model': model, 'loss_track': loss_track, 'best_loss': best_loss, 'best_head': best_head, 'c_loss': c_loss, 'best_loss_head': best_loss_head, 'best_epoch': best_epoch,  'epoch': epoch, 'prefix': prefix, 'rID': rID }
        parameter = evaluate_loss_track(p,pdict)
        best_loss  = parameter['best_loss']
        best_loss_head = parameter['best_loss_head']
        best_epoch     = parameter['best_epoch']
        loss_track     = parameter['loss_track']
       
 
    loss_track.to_csv('EVALUATION/'+rID+'/'+prefix+'_loss_statistics.csv')
        #print('best_epoch: ',best_epoch)
    print('-------------------SESSION COMPLETED--------------------------')
    return evaluation(device_id,p,model,val_loader,start_stats,best_loss,last_loss)


def general_session(rID,components,p,prefix,last_loss,gpu_id=0): 

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
    p['rID'] = rID
    p['prefix'] = prefix
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

    if p['train_split'] in ['train','test']:
        Vdataloader = val_loader['train_split'] if p['train_split'] == 'test' else val_loader['test_split']
    else: Vdataloader = val_loader['val_loader']

    if p['num_heads'] > 1:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,Vdataloader,forwarding='singleHead_eval')
        starting_data.compute_kNN_statistics(100)
        starting_data.compute_real_consistency(0.5)
        start_stats = starting_data.return_statistic_summary(c_loss)
    else:
        best_loss = c_loss
        best_loss_head = best_head
        starting_data = Analysator(device_id,model,Vdataloader)
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
        pdict = {'model': model, 'best_head': best_head, 'c_loss': c_loss, 'best_loss': best_loss,'best_loss_head': best_loss_head, 'best_epoch': best_epoch, 'epoch': epoch, 'prefix': prefix, 'rID': rID }
        parameter = evaluate_standard(p,pdict)
        best_loss  = parameter['best_loss']
        best_loss_head = parameter['best_loss_head']
        best_epoch     = parameter['best_epoch']
       
 
    print('-------------------SESSION COMPLETED--------------------------')
    #print('best_epoch: ',best_epoch)
    return evaluation(device_id,p,model,val_loader,start_stats,best_loss,last_loss)


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
    rID = parameters['rID']

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
                ##add_file_path('/home/blachm86/'+rID+'_files.txt',p['scan_model'])
            else:
                torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')
                ##add_file_path('/home/blachm86/'+rID+'_files.txt','PRODUCTS/'+prefix+'_best_model.pth')
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
            ##add_file_path('/home/blachm86/'+rID+'_files.txt','PRODUCTS/'+prefix+'_best_model.pth')
 

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
    rID = parameters['rID']
    

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
                ##add_file_path('/home/blachm86/'+rID+'_files.txt',p['scan_model'])
            else:
                torch.save(model.state_dict(),'PRODUCTS/'+prefix+'_best_model.pth')
                ##add_file_path('/home/blachm86/'+rID+'_files.txt','PRODUCTS/'+prefix+'_best_model.pth')
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
            ##add_file_path('/home/blachm86/'+rID+'_files.txt','PRODUCTS/'+prefix+'_best_model.pth')
            
    pdict = {}
    pdict['best_loss'] = best_loss
    pdict['best_head'] = best_head
    pdict['c_loss'] = c_loss
    pdict['best_loss_head'] = best_loss_head
    pdict['best_epoch'] = best_epoch
    
    return pdict


def evaluation(device_id,p,model,loaders,start_stats,best_loss,last_loss,model_type='cluster_head'):

    rID = p['rID'] # OK
    prefix = p['prefix'] # OK

    if p['num_heads'] > 1: 
        if p['train_method'] == 'scan':
            print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
            model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
            model.load_state_dict(model_checkpoint['model']) # Ab hier ist model optimal
           
            if p['train_split'] in ['train','test']: # ! for CIFAR10 use split 'cifar'

                train_split_loader = loaders['train_split'] # OK
                test_split_loader = loaders['test_split'] # OK
                train_split_data = Analysator(device_id,model,train_split_loader,forwarding='singleHead_eval',model_type=model_type) # OK
                train_split_data.compute_kNN_statistics(100)
                train_split_data.compute_real_consistency(0.5)
                test_split_data = Analysator(device_id,model,test_split_loader,forwarding='singleHead_eval',model_type=model_type) # OK
                test_split_data.compute_kNN_statistics(100)
                test_split_data.compute_real_consistency(0.5)
                train_split_session = train_split_data.return_statistic_summary(best_loss)
                test_split_session = test_split_data.return_statistic_summary(best_loss)
                session_stats = test_split_session if p['train_split'] == 'train' else train_split_session # OK

                predictions = get_predictions(device_id, p, train_split_loader, model) #if p['train_split'] == 'train'  else  get_predictions(device_id, p, train_split_loader, model)
                clustering_stats = hungarian_evaluate(device_id, model_checkpoint['head'], predictions,
                                                            class_names=train_split_loader.dataset.classes,
                                                            compute_confusion_matrix=True,
                                                            confusion_matrix_file=os.path.join(p['scan_dir'],prefix+'TrainSplit_confusion_matrix.png'))
                predictions = get_predictions(device_id, p, test_split_loader, model)
                clustering_stats = hungarian_evaluate(device_id, model_checkpoint['head'], predictions,
                                                            class_names=test_split_loader.dataset.classes,
                                                            compute_confusion_matrix=True,
                                                            confusion_matrix_file=os.path.join(p['scan_dir'],prefix+'TestSplit_confusion_matrix.png'))
                torch.save({'analysator': train_split_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'TrainSplit_ANALYSATOR')
                torch.save({'analysator': test_split_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'TestSplit_ANALYSATOR')

            else:
                val_loader = loaders['val_loader']
                val_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval',model_type=model_type)
                val_data.compute_kNN_statistics(100)
                val_data.compute_real_consistency(0.5)
                session_stats = val_data.return_statistic_summary(best_loss)
                predictions = get_predictions(device_id, p, val_loader, model) #if p['train_split'] == 'train'  else  get_predictions(device_id, p, train_split_loader, model)
                clustering_stats = hungarian_evaluate(device_id, model_checkpoint['head'], predictions,
                                                            class_names=val_loader.dataset.classes,
                                                            compute_confusion_matrix=True,
                                                            confusion_matrix_file=os.path.join(p['scan_dir'],prefix+'_confusion_matrix.png'))
                torch.save({'analysator': val_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR') # OK
            ##add_file_path('/home/blachm86/'+rID+'_files.txt',str(os.path.join(p['scan_dir'],prefix+'_confusion_matrix.png')))


            ##add_file_path('/home/blachm86/'+rID+'_files.txt','EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')
            if best_loss < last_loss:
                return start_stats ,session_stats, True
            else:
                return start_stats ,session_stats, False

        else:
            #print('best accuracy: ',best_accuracy)
            #print('head_id: ',best_loss_head)
            #print('\n')
            best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
            model.load_state_dict(best_copy)

            if p['train_split'] in ['train','test']:

                train_split_loader = loaders['train_split']
                test_split_loader = loaders['test_split']
                train_split_data = Analysator(device_id,model,train_split_loader,forwarding='singleHead_eval',model_type=model_type)
                train_split_data.compute_kNN_statistics(100)
                train_split_data.compute_real_consistency(0.5)
                test_split_data = Analysator(device_id,model,test_split_loader,forwarding='singleHead_eval',model_type=model_type)
                test_split_data.compute_kNN_statistics(100)
                test_split_data.compute_real_consistency(0.5)

                train_split_session = train_split_data.return_statistic_summary(best_loss)
                test_split_session = test_split_data.return_statistic_summary(best_loss)
                session_stats = test_split_session if p['train_split'] == 'train' else train_split_session # OK

                torch.save({'analysator': train_split_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'TrainSplit_ANALYSATOR')
                torch.save({'analysator': test_split_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'TestSplit_ANALYSATOR')
            
            else:

                val_loader = loaders['val_loader']
                metric_data = Analysator(device_id,model,val_loader,forwarding='singleHead_eval',model_type=model_type)
                metric_data.compute_kNN_statistics(100)
                metric_data.compute_real_consistency(0.5)
                session_stats = metric_data.return_statistic_summary(best_loss)
                torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')


            if best_loss < last_loss:
                return start_stats ,session_stats, True
            else:
                return start_stats ,session_stats, False

    else: 

        best_copy = torch.load('PRODUCTS/'+prefix+'_best_model.pth',map_location='cpu')
        model.load_state_dict(best_copy)
        val_loader = loaders['val_loader']
        metric_data = Analysator(device_id,model,val_loader,forwarding='head',model_type=model_type)
        metric_data.compute_kNN_statistics(100)
        metric_data.compute_real_consistency(0.5)
        session_stats = metric_data.return_statistic_summary(best_loss)
        torch.save({'analysator': metric_data,'parameter':p},'EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')
        ##add_file_path('/home/blachm86/'+rID+'_files.txt','EVALUATION/'+rID+'/'+prefix+'_ANALYSATOR')

        if best_loss < last_loss:
            return start_stats ,session_stats, True
        else:
            return start_stats ,session_stats, False


class statistics_register():

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