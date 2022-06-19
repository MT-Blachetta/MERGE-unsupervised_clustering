import argparse
from utils.config import create_config
from functionality import initialize_training, collate_custom, get_optimizer
from datasets import ReliableSamplesSet
import torchvision.transforms as transforms
from utils.common_config import adjust_learning_rate
from evaluate import Analysator
import torchvision
import numpy as np
#from testmodule import TEST_initial_model
import torch
import copy
from evaluate import get_cost_matrix, assign_classes_hungarian, accuracy_from_assignment


#@ref=[main_command_line]
FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier', default=0)
FLAGS.add_argument('-prefix',help='prefix name')
FLAGS.add_argument('-root',help='root directory')
FLAGS.add_argument('-config', help='path to the model files')

args = FLAGS.parse_args()

print('arguments.gpu = ',args.gpu)
print('arguments.prefix = ',args.prefix)
print('arguments.config = ',args.config)
print('arguments.root = ',args.root)

#!@ref

p = create_config(args.root, args.config, args.prefix)

p['train_args'] = {}
p['train_args']['device'] = 'cuda'
p['train_args']['gpu_id'] = args.gpu
p['device'] = 'cuda:'+str(args.gpu)
#p['prefix'] = args.prefix
#p['gpu_id'] = args.gpu
num_cluster = p['num_classes']
fea_dim = p['feature_dim']
p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]

#with open('p_info.txt','w') as f:
#    f.write(str(p))

params = initialize_training(p)

dataset = params['dataset'] # OK ; 0:[%bhbx,%fas]
model = params['model']
criterion = params['criterion']
optimizer = params['optimizer']
train_epoch = params['train_method']
val_loader = params['val_dataloader']
train_model = torchvision.models.resnet18() # %tv_res18
train_model.fc = torch.nn.Linear(512,p['num_classes']) # %tv_res18
train_optimizer = get_optimizer(p,train_model) 

#print('-----MODEL ANALYSIS-----')
#with open('model_analysis','w') as f: f.write(str(model))
#print('model_backbone',type(model.backbone))
#print('model_backbone_dim',model.backbone_dim)
#print('model_nheads',model.nheads)
#print('model_aug_type',model.aug_type)
#print('model_head',type(model.head))

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
            


val_transformations = transforms.Compose([
                            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                            transforms.ToTensor(),
                            transforms.Normalize(**p['transformation_kwargs']['normalize'])])

dataset.transform = val_transformations

#TEST_initial_model(model,dataset,val_transformations)

#training_set = ReliableSamplesSet(dataset,val_transformations)
#training_set.evaluate_samples(p,model)
batch_loader = torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
                                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                drop_last=True, shuffle=True)


print('start training loop...')
for epoch in range(0, p['epochs']):
    
    print('\nepoch: ',epoch)

    
    lr = adjust_learning_rate(p, train_optimizer, epoch)
    print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
    print('Train ...')
    train_epoch(batch_loader, train_model, criterion, train_optimizer, epoch, p['train_args'],True)
    
    #@COMPONENT:Evaluation&Measures
    #val_dataset = copy.deepcopy(dataset)
    #val_dataset.transform = val_transformations
    #val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=p['num_workers'], batch_size=p['batch_size'], pin_memory=True,collate_fn=collate_custom, drop_last=False, shuffle=False)
    #metric_data = Analysator(p['device'],train_model,val_loader)
    compute_accuracy(p['device'],train_model,val_loader)

    #print('Accuracy: ',metric_data.get_accuracy())
    #!@

    # TO DO:
    
    # Checkpoint
    # Evaluation

#@COMPONENT:Evaluation&Measures
#val_dataset = dataset
#val_dataset.transform = val_transformations
#val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=p['num_workers'], batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom, drop_last=False, shuffle=False)
#metric_data = Analysator(p['device'],model,val_loader)
#torch.save({'analysator': metric_data,'parameter':p},'SELFLABEL/'+args.prefix+'_ANALYSATOR')
#!@

torch.save(train_model.state_dict(),'SELFLABEL/'+args.prefix+'_model.pth')