import argparse
import os
import torch
from torch._six import string_classes
int_classes = int
from utils.config import create_config
from evaluate import logger
import pandas as pd
from functionality import initialize_training
from execution_utils import loss_track_session, general_session, statistics_register

#@author: Michael Blachetta

FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier')
FLAGS.add_argument('-config',help='path to the trial list')
FLAGS.add_argument('-p',help='prefix')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')
#FLAGS.add_argument('--model_path', help='path to the model files')
FLAGS.add_argument('-loss_track',help='wether to track loss value',default='no')

def main():
    args = FLAGS.parse_args()

    logging = logger({'args.p':str(args.p),'args.root_dir':str(args.root_dir),'args.config':str(args.config),'args.loss_track':str(args.loss_track)})

    p = create_config(args.p ,args.root_dir, args.config, args.p)
    prefix = args.p
    p['prefix'] = args.p
    gpu_id = args.gpu
    num_cluster = p['num_classes']
    fea_dim = p['feature_dim']
    p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
    params = initialize_training(p)

    rlog = logger(value=p,unit_name=str(args.p),unit_type='<Session>')
    rlog2 = logger(value={'objects': params_to_typestring(params)},unit_name='programm_components',unit_type='datatypes:')
    rlog.add_element(rlog2)
    logging.add_element(rlog)
    
    last_loss = 1000

    if args.loss_track in ['y','yes']:
        start_info, end_info, new_optimum = loss_track_session(prefix,params,p,prefix,last_loss,gpu_id)
    else:
        start_info, end_info, new_optimum = general_session(prefix,params,p,prefix,last_loss,gpu_id)

    with open('EVALUATION/'+prefix+'/'+prefix+'_log.txt','w') as f:
        f.write(str(logging))
        f.write('\n--------------RESULTS--------------\n')
        f.write(str(end_info))
       

    session_stats = statistics_register()
    session_stats.add_session_statistic(0,p,start_info,end_info)
    torch.save(session_stats,'EVALUATION/'+prefix+'/'+prefix+'.session')


    with open('/home/blachm86/'+args.p+'_files.txt','w') as f:
        f.write('['+'"EVALUATION/'+prefix+'/'+prefix+'_log.txt",')
        f.write('"EVALUATION/'+prefix+'/'+prefix+'.session",')
        
        if args.loss_track in ['y','yes']: f.write('"EVALUATION/'+prefix+'/'+prefix+'_loss_statistics.csv",')

        if p['train_split'] in ['train','test']:
            f.write('"EVALUATION/'+prefix+'/'+prefix+'TrainSplit_ANALYSATOR",')
            f.write('"EVALUATION/'+prefix+'/'+prefix+'TestSplit_ANALYSATOR",')

            if p['train_method'] == "scan":           
                f.write('"'+os.path.join(p['scan_dir'],prefix+'TestSplit_confusion_matrix.png')+'",')
                f.write('"'+os.path.join(p['scan_dir'],prefix+'TrainSplit_confusion_matrix.png')+'",')
                f.write('"'+p['scan_model']+'"')
            else: f.write('"PRODUCTS/'+prefix+'_best_model.pth"')

        else:
            f.write('"EVALUATION/'+prefix+'/'+prefix+'_ANALYSATOR",')
            if p['train_method'] == "scan":
                f.write('"'+str(os.path.join(p['scan_dir'],prefix+'_confusion_matrix.png'))+'",')
                f.write('"'+p['scan_model']+'"')
            else: f.write('"PRODUCTS/'+prefix+'_best_model.pth"')
        f.write(']')
        

def params_to_typestring(para_dict,separator='; '):

    out = ''
    for k in para_dict.keys():
        out += k+'='+str(type(para_dict[k]))+separator

    return out

if __name__ == "__main__":
    main()
