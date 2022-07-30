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
from functionality import get_backbone, get_head_model, get_augmentation ,get_dataset, get_val_dataloader, get_direct_valDataloader, get_optimizer

#@ref=[main_command_line]
FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier', default=0)
FLAGS.add_argument('-config', help='path to the model files')
FLAGS.add_argument('-prefix',help='prefix name')
FLAGS.add_argument('-root',help='root directory',default='SELFLABEL')


def main(args):

    #print('arguments.gpu = ',args.gpu)
    #print('arguments.prefix = ',args.prefix)
    #print('arguments.config = ',args.config)
    #print('arguments.root = ',args.root)

    p = create_config(args.prefix,args.root, args.config, args.prefix)
    p['device'] = 'cuda:'+str(args.gpu)
    p['prefix'] = args.prefix
    p['gpu_id'] = args.gpu
    num_cluster = p['num_classes']
    fea_dim = p['feature_dim']
    p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]

    dataset = get_dataset(p,None)
    val_loader = get_val_dataloader(p)

    strong_transform = get_augmentation(p)   
    weak_transform = get_augmentation(p,aug_method='standard',aug_type='scan')
    val_transform = transforms.Compose([
                            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                            transforms.ToTensor(),
                            transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    medium_transform = get_augmentation(p,aug_method='simclr',aug_type='scan')

    from datasets import pseudolabelDataset, fixmatchDataset
    unlabeled_data = fixmatchDataset(dataset,weak_transform,strong_transform)
    reliable_samples = torch.load(p['indexfile_path'],map_location='cpu')
    labeled_data = pseudolabelDataset(dataset, reliable_samples['sample_index'], reliable_samples['pseudolabel'], val_transform, medium_transform)
    base_dataloader = get_direct_valDataloader(p)

    backbone = get_backbone(p,secondary=True) # OK
    model = get_head_model(p,backbone,secondary=True) # OK

    model_dict = torch.load(p['fixmatch_model']['pretrain_path'],map_location='cpu')
    ltext = model.load_state_dict(model_dict,strict=True)
    print(ltext)
        #model_type = 'fixmatch_model'

    from models import MlpHeadModel
    model_args = p['model_args']
    use_model = MlpHeadModel(model.resnet,model.resnet.rep_dim,model_args)
    model_type = 'cluster_head'


    optimizer = get_optimizer(p,model)

    #labeled_data.evaluate_samples(p,pretrain_model)

    label_step = len(labeled_data)//p['batch_size']
    unlabeled_step = len(unlabeled_data)//p['batch_size']
    step_size = min(label_step,unlabeled_step)

    labeled_loader = torch.utils.data.DataLoader(labeled_data, num_workers=p['num_workers'], 
                                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                drop_last=True, shuffle=True)

    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_data, num_workers=p['num_workers'], 
                                                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                drop_last=True, shuffle=True)

    val_dataloader = val_loader['val_loader']

    print('start training loop...')
    for epoch in range(0, p['epochs']):
        
        print('\nepoch: ',epoch)
        #print('dataset_len = ',len(training_set))
        consistency_tensor = compute_consistency(p['device'],use_model,base_dataloader,200,model_type)
        #consistency_tensor.detach().cpu()
        #model.to('cpu')

        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

            # Train
        print('Train ...')
        fixmatch_train(p['device'],use_model,labeled_loader,unlabeled_loader,consistency_tensor,optimizer,step_size,threshold=p['confidence_threshold'],temperature=p['temperature'],lambda_u=p['lambda_u'],augmented=True)
        
        #@COMPONENT:Evaluation&Measures
        #val_dataset = copy.deepcopy(dataset)
        #val_dataset.transform = val_transformations
        #val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=p['num_workers'], batch_size=p['batch_size'], pin_memory=True,collate_fn=collate_custom, drop_last=False, shuffle=False)
        #metric_data = Analysator(p['device'],model,val_loader)
        #print('Accuracy: ',metric_data.get_accuracy())

        compute_accuracy(p['device'],use_model,val_dataloader)


    final_accuracy = compute_accuracy(p['device'],model,val_dataloader)

   # with open('SELFLABEL/'+args.prefix+'_log.txt','w') as f:
   #     f.write(str(logging))
   #     f.write('\nAccuracy = '+str(final_accuracy))

    torch.save(model.state_dict(),'SELFLABEL/'+args.prefix+'_model.pth')

def compute_consistency(device,model,base_dataloader,kNN=200,model_type='fixmatch_model',forwarding='head'): # execute for each epoch
        

        model.eval()
        predictions = []
        features = []
        soft_labels = []
        confidences = []

        model = model.to(device) # OK(%-cexp_00)
        #print('MODEL first critical expresssion: ',type(model))



        with torch.no_grad():      
            for batch in base_dataloader:
                if isinstance(batch,dict):
                    image = batch['image']
                else:
                    image = batch[0]

                if model_type == 'contrastive_clustering':
                    image = image.to(device)
                    feats, _,preds, _ = model(image,image)
    
                elif model_type == 'fixmatch_model':
                    image = image.to(device)
                    feats = model(image,forward_pass='features')
                    preds = model(image)
                    
                else:
                    image = image.to(device)
                    feats = model(image,forward_pass='features')
                    preds = model(feats,forward_pass=forwarding)


                features.append(feats)
                soft_labels.append(preds)
                max_confidence, prediction = torch.max(preds,dim=1) 
                predictions.append(prediction)
                confidences.append(max_confidence)


            feature_tensor = torch.cat(features)
                #self.softlabel_tensor = torch.cat(soft_labels)
            self_predictions = torch.cat(predictions)
                #print('max_prediction A: ',self.predictions.max())
            self_predictions = self_predictions.type(torch.LongTensor)
                #print('max_prediction B: ',self.predictions.max())
                #print('len(self.predictions) B: ',len(self.predictions))
            #self_num_clusters = self_predictions.max()+1 # !issue: by test config assert(self.num_clusters == 10) get 9
                #print('num_clusters: ',self.num_clusters)
            #self_confidence = torch.cat(confidences)
            #dataset_size = len(self_predictions)

            feature_tensor = torch.nn.functional.normalize(feature_tensor, dim = 1)

            idx_list = []
            for i in range(len(feature_tensor)):
                feature = torch.unsqueeze(feature_tensor[i],dim=0)
                similarities = torch.mm(feature,feature_tensor.t())
                scores, idx_ = similarities.topk(k=kNN, dim=1)
                idx_list.append(idx_)
            idx_k = torch.cat(idx_list)

            labels_topk = torch.zeros_like(idx_k)
            #confidence_topk = torch.zeros_like(idx_k,dtype=torch.float)
            for s in range(kNN):
                labels_topk[:, s] = self_predictions[idx_k[:, s]]
                #confidence_topk[:, s] = self_confidence[idx_k[:, s]]
            
            kNN_consistent = labels_topk[:, 0:1] == labels_topk # <boolean mask>
            #kNN_labels = labels_topk
            consistencies = kNN_consistent.sum(dim=1)/kNN
            #assert(len(consistencies) == 100000) <-------------- !
            print('Consistency_tensor SHAPE = ',consistencies.shape)

            return consistencies




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


if __name__ == '__main__':
    args = FLAGS.parse_args()
    main(args)
