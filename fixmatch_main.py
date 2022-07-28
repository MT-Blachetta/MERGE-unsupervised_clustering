import argparse
from utils.config import create_config
from datasets import ReliableSamplesSet
import torchvision.transforms as transforms
from functionality import collate_custom
from utils.common_config import adjust_learning_rate
from evaluate import Analysator, logger
#from testmodule import TEST_initial_model
import torch
import copy
import numpy as np
from evaluate import get_cost_matrix, assign_classes_hungarian, accuracy_from_assignment
from training import fixmatch_train
from functionality import initialize_fixmatch_training

#@ref=[main_command_line]
FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier', default=0)
FLAGS.add_argument('-config', help='path to the model files')
FLAGS.add_argument('-prefix',help='prefix name')
FLAGS.add_argument('-root',help='root directory',default='SELFLABEL')


def main():
    args = FLAGS.parse_args()

    #print('arguments.gpu = ',args.gpu)
    #print('arguments.prefix = ',args.prefix)
    #print('arguments.config = ',args.config)
    #print('arguments.root = ',args.root)

    p = create_config(args.prefix,args.root, args.config, args.prefix)
    #p['train_args'] = {}
    #p['train_args']['device'] = 'cuda'
    #p['train_args']['gpu_id'] = args.gpu
    p['device'] = 'cuda:'+str(args.gpu)
    p['prefix'] = args.prefix
    #p['gpu_id'] = args.gpu
    #num_cluster = p['num_classes']
    #fea_dim = p['feature_dim']
    #p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]
    params = initialize_fixmatch_training(p)

    logging = logger({'args.prefix':str(args.prefix),'args.root':str(args.root),'args.config':str(args.config)},'SELFLABEL_Method')
    rlog = logger(value=p,unit_name=str(args.p),unit_type='<Session>')
    param_types = {k: str(type(params[k])) for k in params.keys() }
    rlog2 = logger(value=param_types,unit_name='programm_components',unit_type='datatypes:')
    rlog.add_element(rlog2)
    logging.add_element(rlog)

    labeled_dataloader = params['label_dataloader'] 
    unlabeled_dataset = params['unlabeled_dataset'] 
    val_loader = params['validation_loader'] 
    step_size = params['step_size'] 
    optimizer = params['optimizer'] 
    model = params['model'] 
    model_type = params['model_type']


    print('start training loop...')
    for epoch in range(0, p['epochs']):
        
        print('\nepoch: ',epoch)
        #print('dataset_len = ',len(training_set))
        unlabeled_dataset.compute_consistency(p,model,model_type)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, num_workers=p['num_workers'], 
                                                    batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                    drop_last=True, shuffle=True)

        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

            # Train
        print('Train ...')
        fixmatch_train(p['device'],model,labeled_dataloader,unlabeled_loader,optimizer,step_size,threshold=p['confidence_threshold'],temperature=p['temperature'],lambda_u=p['lambda_u'],augmented=True)
        
        #@COMPONENT:Evaluation&Measures
        #val_dataset = copy.deepcopy(dataset)
        #val_dataset.transform = val_transformations
        #val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=p['num_workers'], batch_size=p['batch_size'], pin_memory=True,collate_fn=collate_custom, drop_last=False, shuffle=False)
        #metric_data = Analysator(p['device'],model,val_loader)
        #print('Accuracy: ',metric_data.get_accuracy())
        compute_accuracy(p['device'],model,val_loader)

    final_accuracy = compute_accuracy(p['device'],model,val_loader)

    with open('SELFLABEL/'+args.prefix+'_log.txt','w') as f:
        f.write(str(logging))
        f.write('\nAccuracy = '+str(final_accuracy))

    torch.save(model.state_dict(),'SELFLABEL/'+args.prefix+'_model.pth')





def compute_accuracy(device,model,loader):
        
    model.eval()
    model = model.to(device) # OK(%-cexp_00)
    predictions = []
    labels = []
    softmax_fn = torch.nn.Softmax(dim = 1)

    with torch.no_grad():      
        for batch in loader:
            if isinstance(batch,dict):
                image = batch['image']
                label = batch['target']
            else:
                image = batch[0]
                label = batch[1]

            image = image.to(device,non_blocking=True)
            preds = model(image)
            max_confidence, predict = torch.max(softmax_fn(preds),dim=1) 
            predictions.append(predict)
            labels.append(label)

    yt = torch.cat(labels)
    pr = torch.cat(predictions)
    y_train = np.array(yt.detach().cpu().numpy())
    pred = np.array(pr.detach().cpu().numpy())
    max_label = max(y_train)
    C = get_cost_matrix(pred, y_train, max_label+1)
    ri, ci = assign_classes_hungarian(C)
    accuracy = accuracy_from_assignment(C,ri,ci)
    print('PERFORMANCE: ',accuracy)

    return accuracy

