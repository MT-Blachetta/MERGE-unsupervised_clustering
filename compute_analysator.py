import torch
import torch.nn as nn
import torch.nn.functional as F
from scatnet import ScatSimCLR
from evaluate import Analysator
import torchvision.transforms as transforms
import copy

class MLP(nn.Module): # id MlpHead
    def __init__(self, num_neurons, batch_norm=True):
        super(MLP, self).__init__()
        num_layer = len(num_neurons) - 1
        for i in range(num_layer):
            layer_name = "lin{}".format(i+1)
            layer = nn.Linear(num_neurons[i], num_neurons[i+1])
            self.add_module(layer_name, layer)

            if batch_norm:
                layer_name = "bn{}".format(i+1)
                layer = nn.BatchNorm1d(num_neurons[i+1])
                self.add_module(layer_name, layer)

        self.num_layer = num_layer
        self.batch_norm = batch_norm


    def forward(self, x):
        num_layer = self.num_layer

        for i in range(num_layer):
            layer_name = "lin{}".format(i+1)
            layer = self.__getattr__(layer_name)
            x = layer(x)

            if self.batch_norm:
                bn_name = "bn{}".format(i+1)
                bn = self.__getattr__(bn_name)
                x = bn(x)

            if i < num_layer - 1:
                x = F.relu(x, inplace=True)

        return x


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        #self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        self.cluster_head = nn.ModuleList([MLP([self.backbone_dim,self.backbone_dim,nclusters]) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out

class MLP_head_model(nn.Module):
    def __init__(self,backbone,head):
        super(MLP_head_model, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self,x,forward_pass = 'default'):

        if forward_pass == 'default':
            features = self.backbone(x)
            return self.head(features)

        elif forward_pass == 'backbone':
            return self.backbone(x)

        elif forward_pass == 'head':
            return self.head(x)


# build scatnet backbone
backbone = ScatSimCLR(J=2, L=16, input_size=(96, 96, 3), res_blocks=30, out_dim=128)
backbone_dict = {'backbone': backbone, 'dim': 128}

# wrap it into ClusteringModel

model = ClusteringModel(backbone_dict,10,10)

scan_save = torch.load('/home/blachm86/SCAN/RESULTS/stl-10/scan/scatnet_model.pth.tar',map_location='cpu')
itext = model.load_state_dict(scan_save['model'],strict=True)
print('itext: ',itext)

best_head = copy.deepcopy(model.cluster_head[scan_save['head']])
print('best_head: ',best_head)
torch.save(best_head.state_dict(),'scan_transfer_head.pth')


eval_model = MLP_head_model(model.backbone,best_head)

# now get dataset with dataloader

print('get validation dataset')
    
    # dataset:
from datasets import STL10_eval
from functionality import collate_custom


val_transformations = transforms.Compose([
                                transforms.CenterCrop(96),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
eval_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)


val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=8,
                batch_size=256, pin_memory=True, collate_fn=collate_custom,
                drop_last=False, shuffle=False)

print('compute features in Analysator')

eval_object = Analysator('cuda:3',eval_model,val_dataloader)

eval_object.compute_kNN_statistics(100)
eval_object.compute_real_consistency(0.5)
eval_object.return_statistic_summary(0)

torch.save(eval_object,'scan_analysator.torch')




