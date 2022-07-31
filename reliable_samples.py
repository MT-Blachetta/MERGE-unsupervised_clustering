from datasets import STL10
import torchvision.transforms as transforms
from scatnet import ScatSimCLR
import torch
import torch.nn as nn
from functionality import collate_custom, get_backbone, get_head_model



split = 'train+unlabeled'
dataset_id = 'stl-10'
pretrain_path = '/home/blachm86/backbone_models/cc_stl10.tar'
model_type = 'fixmatch_model'
device = 'cuda:0'
samples_per_class = 500

kNN = 200
num_classes = 10


# Dataset

val_transform = transforms.Compose([transforms.CenterCrop(96), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

if dataset_id == 'cifar-10':
    from datasets import CIFAR10
    dataset = CIFAR10(train=True, transform=val_transform, download=True)
            #eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

elif dataset_id == 'cifar-20':
    from datasets import CIFAR20
    dataset = CIFAR20(train=True, transform=val_transform, download=True)
        #eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)

elif dataset_id == 'stl-10':
    if split in ['train', 'test','unlabeled','train+unlabeled']:
        dataset = STL10(split=split, transform=val_transform, download=False)
    else:
        from datasets import STL10_eval
        dataset = STL10_eval(path='/space/blachetta/data',aug=val_transform)


dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=512, pin_memory=True, collate_fn=collate_custom, drop_last=False, shuffle=False)


# Model

p = {'num_classes': 10, 'backbone': 'ResNet34', 'pretrain_type': 'fixmatch', 'pretrain_path': pretrain_path, 'feature_dim': 128, 'hidden_dim': 128, 'scatnet_args': { 'J': 2, 'L': 16, 'input_size': [96, 96, 3] , 'res_blocks': 30, 'out_dim': 128 },
'num_heads': 10, 'model_type': 'fixmatch', 'model_args':{'head_type': 'mlp','aug_type': 'default','batch_norm': False, 'last_batchnorm': False, 'last_activation': 'None', 'drop_out': -1 } }

num_cluster = p['num_classes']
fea_dim = p['feature_dim']
p['model_args']['num_neurons'] = [fea_dim, fea_dim, num_cluster]



class MLP_head_model(nn.Module):
    def __init__(self,backbone,head):
        super(MLP_head_model, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self,x,forward_pass = 'default'):

        if forward_pass == 'default':
            features = self.backbone(x)
            return torch.nn.functional.softmax(self.head(features),dim=1)

        elif forward_pass == 'features':
            return self.backbone(x)

        elif forward_pass == 'head':
            return torch.nn.functional.softmax(self.head(x),dim=1)

        else: ValueError('invalid forward pass')

backbone = get_backbone(p)
#ScatSimCLR(J=2, L=16, input_size=(96, 96, 3), res_blocks=30, out_dim=128)
#backbone_dict = {'backbone': backbone, 'dim': 128}
model = get_head_model(p,backbone)

#scan_save = torch.load(pretrain_path,map_location='cpu')
#itext = model.load_state_dict(scan_save['model'])
#print('itext: ',itext)

model_save = torch.load(pretrain_path,map_location='cpu')
itext = model.load_state_dict(model_save)
print('itext: ',itext)

#best_head = model.cluster_head[scan_save['head']]
#print('best_head: ',best_head)
#torch.save(best_head.state_dict(),'scan_transfer_head.pth')

#eval_model = MLP_head_model(model.backbone,best_head)
eval_model = model

eval_model.eval()
predictions = []
features = []
confidences = []
indices = []
eval_model = eval_model.to(device)

with torch.no_grad():      
    for batch in dataloader:

        image = batch['image']
        index = batch['meta']['index']


        if model_type == 'contrastive_clustering':
            image = image.to(device,non_blocking=True)
            feats, _,preds, _ = eval_model(image,image)
    
        elif model_type == 'fixmatch_model':
            image = image.to(device,non_blocking=True)
            feats = eval_model(image,forward_pass='features')
            preds = eval_model(image)
                    
        else:
            image = image.to(device,non_blocking=True)
            feats = eval_model(image,forward_pass='features')
            preds = eval_model(feats,forward_pass='head')


        features.append(feats)

        indices.append(index)
        max_confidence, prediction = torch.max(preds,dim=1) 
        predictions.append(prediction)
        confidences.append(max_confidence)


feature_tensor = torch.cat(features)
index_tensor = torch.cat(indices)
prediction_tensor = torch.cat(predictions)
confidence_tensor = torch.cat(confidences)
confidences = confidence_tensor

label_selections = []
print(index_tensor)


feature_tensor = torch.nn.functional.normalize(feature_tensor, dim = 1)

idx_list = []
for i in range(len(feature_tensor)):
    feature = torch.unsqueeze(feature_tensor[i],dim=0)
    similarities = torch.mm(feature,feature_tensor.t())
    scores, idx_ = similarities.topk(k=kNN, dim=1)
    idx_list.append(idx_)
idx_k = torch.cat(idx_list)

labels_topk = torch.zeros_like(idx_k)
confidence_topk = torch.zeros_like(idx_k,dtype=torch.float)
for s in range(kNN):
    labels_topk[:, s] = prediction_tensor[idx_k[:, s]]
    confidence_topk[:, s] = confidences[idx_k[:, s]]
            
kNN_consistent = labels_topk[:, 0:1] == labels_topk # <boolean mask>
kNN_confidences = confidence_topk
criterion_consistent = []
for i in range(len(prediction_tensor)):
    confids = kNN_confidences[i][kNN_consistent[i]] # +logical_index > +true for index of consistent label; +size=knn > +indexes topk instances
    real = confids > 0.5
    criterion_consistent.append(sum(real)/kNN)

consistencies = torch.Tensor(criterion_consistent)
# alternative_consistency = kNN_consistent.sum(dim=1)/knn
confidences = confidences.cpu()
performance = 1.5*confidences + consistencies

selection = []

for c in range(num_classes):
    class_indices = torch.where(prediction_tensor == c)[0]
    class_performance = performance[class_indices]
    sorted, sorted_indices = torch.sort(class_performance,descending=True)
    si =  sorted_indices[:samples_per_class]
    top_class_indices = class_indices[si]
    print('class ',c,' performance: ',min(sorted[:samples_per_class]))
    selection.append(top_class_indices)

label_index = torch.cat(selection)

#torch.index_select()

selected_predictions = prediction_tensor[label_index]

torch.save({'sample_index': label_index, 'pseudolabel': selected_predictions},'/home/blachm86/train&unlabeled_5%.ind')