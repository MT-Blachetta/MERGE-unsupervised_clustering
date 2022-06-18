import argparse
from utils.config import create_config
from functionality import initialize_training, collate_custom, get_optimizer
from datasets import ReliableSamplesSet
import torchvision.transforms as transforms
from utils.common_config import adjust_learning_rate
from evaluate import Analysator
import torchvision

#from testmodule import TEST_initial_model
import torch
import copy

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
train_model = torchvision.model.resnet18()
train_model.fc = torch.nn.Linear(512,p['num_classes'])
train_optimizer = get_optimizer(p,train_model)

#print('-----MODEL ANALYSIS-----')
#with open('model_analysis','w') as f: f.write(str(model))
#print('model_backbone',type(model.backbone))
#print('model_backbone_dim',model.backbone_dim)
#print('model_nheads',model.nheads)
#print('model_aug_type',model.aug_type)
#print('model_head',type(model.head))


val_transformations = transforms.Compose([
                            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                            transforms.ToTensor(),
                            transforms.Normalize(**p['transformation_kwargs']['normalize'])])

#TEST_initial_model(model,dataset,val_transformations)

training_set = ReliableSamplesSet(dataset,val_transformations)
training_set.evaluate_samples(p,model)
batch_loader = torch.utils.data.DataLoader(training_set, num_workers=p['num_workers'], 
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
    metric_data = Analysator(p['device'],train_model,val_loader)
    print('Accuracy: ',metric_data.get_accuracy())
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

#torch.save(model.state_dict(),'SELFLABEL/'+args.prefix+'_model.pth')