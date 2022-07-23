import argparse
import os
import torch
from torch._six import string_classes
int_classes = int
from utils.config import create_config
from evaluate import  logger
import pandas as pd
from execution_utils import loss_track_session, general_session, statistics_register
from functionality import initialize_training


FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier')
FLAGS.add_argument('-list',help='path to the trial list')
FLAGS.add_argument('-rID',help='runtime ID')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')
#FLAGS.add_argument('--config_exp', help='Location of experiments config file')
#FLAGS.add_argument('--model_path', help='path to the model files') # USE THE FULL PATH IN THE CONFIG FILE !


def main():

    #session_list = ['scan_scatnet','scan_clPcl'] # ,'twist',...]
    #os.path.join

    args = FLAGS.parse_args()
    logging = logger({'args.list':str(args.list),'args.riD':str(args.rID)})


    outfiles = {'EVALUATION/'+args.rID+'/settings.txt'}
    
    with open(args.list,'r') as lf:
        pstr = lf.read()
        session_list = eval(pstr.strip(' \n'))

    logging.properties['list_file'] = str(session_list)

    for session in session_list:
        config_file = 'config/'+session+'.yml'
        trials_file = 'config/'+session+'.py'

        with open(trials_file, 'r') as f:
            parsing_text = f.read()
            trial_list = eval(parsing_text.strip(' \n'))

        p = create_config(args.rID ,args.root_dir, config_file, session)
        prefix = session #
        p['prefix'] = session
        gpu_id = args.gpu
        num_cluster = p['num_classes']
        fea_dim = p['feature_dim']
        p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
        p['config_file'] = config_file
        p['trial_list_file'] = trials_file
        #backbone_file = p['pretrain_path']
        #p['pretrain_path'] = os.path.join(args.model_path,backbone_file)

        i = 0
        run_id = prefix

        s_log = logger(value=p,unit_name=session,unit_type='<Session>')
        s_log.add_head_text(logging.head_str(len(s_log))) # not needed

        last_loss = 1000

        session_stats = statistics_register()

        for trial in trial_list:
            # code per trial:
            run_id = prefix+str(i)
            p = map_parameters(p,trial) # session needs to override p['scan_model'] OK
            params = initialize_training(p)
            vls = {'trial_configuration': str(trial),'train_parameter': params_to_typestring(params) }

            t_log = logger(value=vls,unit_name=run_id,unit_type='<training_process>')
            t_log.add_value('run_id',run_id)
            s_log.add_element(t_log)

            t_log.add_head_text(logging.head_str(len(t_log)))
            t_log.add_head_text(s_log.head_str(t_log))
            

            if trial['save_data']:
                start_info, end_info, new_optimum = loss_track_session(args.rID,params,p,run_id,last_loss,gpu_id)
                outfiles.add('EVALUATION/'+args.rID+'/'+run_id+'_loss_statistics.csv')
                if new_optimum: last_loss = end_info['loss']
                if p['train_method'] == 'scan':
                    t_log.to_file(p['scan_dir']+'/'+run_id+'_log.txt','unit_str')
                    outfiles.add(p['scan_dir']+'/'+run_id+'_log.txt')
                    ##add_file_path('/home/blachm86/'+args.rID+'_files.txt',p['scan_dir']+'/'+run_id+'_log.txt')
                else:
                    t_log.to_file('PRODUCTS/'+run_id+'_log.txt','unit_str')
                    outfiles.add('PRODUCTS/'+run_id+'_log.txt')
                    ##add_file_path('/home/blachm86/'+args.rID+'_files.txt','PRODUCTS/'+run_id+'_log.txt')
                
            else:
                start_info, end_info, new_optimum = general_session(args.rID,params,p,run_id,last_loss,gpu_id)
                if new_optimum: 
                    last_loss = end_info['loss']
                    if p['train_method'] == 'scan':  
                        t_log.to_file(p['scan_dir']+'/'+run_id+'_log.txt','unit_str')
                        outfiles.add(p['scan_dir']+'/'+run_id+'_log.txt')
                    else: 
                        t_log.to_file('PRODUCTS/'+run_id+'_log.txt','unit_str' )
                        outfiles.add('PRODUCTS/'+run_id+'_log.txt')
                        ##add_file_path('/home/blachm86/'+args.rID+'_files.txt','PRODUCTS/'+run_id+'_log.txt' )
                else: # delete files
                    if p['train_method'] == 'scan':  
                        if os.path.exists(p['scan_model']): os.remove(p['scan_model'])
                    else:
                        if os.path.exists('PRODUCTS/'+run_id+'_best_model.pth'): os.remove('PRODUCTS/'+run_id+'_best_model.pth')
            # compute table row statistics
            i += 1

            session_stats.add_session_statistic(i,p,start_info,end_info)
            #outfiles.add('EVALUATION/'+args.rID+'/'+run_id+'_ANALYSATOR')
            if p['train_split'] in ['train','test']:
                outfiles.add('EVALUATION/'+args.rID+'/'+run_id+'TrainSplit_ANALYSATOR,')
                outfiles.add('EVALUATION/'+args.rID+'/'+run_id+'TestSplit_ANALYSATOR,')            
                if p['train_method'] == 'scan':
                    outfiles.add(os.path.join(p['scan_dir'],run_id+'TestSplit_confusion_matrix.png'))
                    outfiles.add(os.path.join(p['scan_dir'],run_id+'TrainSplit_confusion_matrix.png'))
                    outfiles.add(p['scan_model'])
                else: outfiles.add('PRODUCTS/'+run_id+'_best_model.pth')
            else:
                outfiles.add('EVALUATION/'+args.rID+'/'+run_id+'_ANALYSATOR')
                if p['train_method'] == 'scan':                    
                    outfiles.add(p['scan_model'])
                    outfiles.add(str(os.path.join(p['scan_dir'],run_id+'_confusion_matrix.png')))
                else: outfiles.add('PRODUCTS/'+run_id+'_best_model.pth')
            #added = pd.DataFrame(dict(next_row),columns=next_row.index,index=[])
            

        logging.add_element(s_log)


        with open('EVALUATION/'+args.rID+'/'+session+'_log.txt','w') as f:
            f.write(s_log.full_str()+'--------------------RESULTS--------------------\n')
            f.write(str(session_stats.output_table))
            #add_file_path('/home/blachm86/'+args.rID+'_files.txt','EVALUATION/'+args.rID+'/'+session+'_log.txt')

        torch.save(session_stats,'EVALUATION/'+args.rID+'/'+prefix+'.session')
        #add_file_path('/home/blachm86/'+args.rID+'_files.txt','EVALUATION/'+args.rID+'/'+prefix+'.session')

        outfiles.add('EVALUATION/'+args.rID+'/'+session+'_log.txt')
        outfiles.add('EVALUATION/'+args.rID+'/'+prefix+'.session')

    with open('EVALUATION/'+args.rID+'/settings.txt','w') as f:
        f.write(str(logging))
        #add_file_path('/home/blachm86/'+args.rID+'_files.txt','EVALUATION/'+args.rID+'/settings.txt')

    with open('/home/blachm86/'+args.rID+'_files.txt','w') as f:
        f.write('[')
        for path in outfiles:
            f.write(str(path)+',')
        f.write(']')


def map_parameters(p,params):
    
    for attribute in params.keys():
        if '.' in attribute:
            primary, secondary = attribute.split('.')
            p[primary][secondary] = params[attribute]
        else:
            p[attribute] = params[attribute]
        
        num_cluster = p['num_classes']
        fea_dim = p['feature_dim']
        p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
        #backbone_file = p['pretrain_path']
        #p['pretrain_path'] = os.path.join(model_path,backbone_file)

        return p


def params_to_typestring(para_dict,separator='; '):

    out = ''
    for k in para_dict.keys():
        out += k+'='+str(type(para_dict[k]))+separator

    return out


if __name__ == "__main__":
    main()