import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralModel(nn.Module):
    def __init__(self,backbone,head):
        super(GeneralModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self,x,**kwargs):
        x = self.backbone.forward(x,**kwargs)
        x = self.head.forward(x,**kwargs)
        return x 


class MLP(nn.Module): # id MlpHead
    def __init__(self, num_neurons, drop_out=-1, last_activation="None", return_extra_index=[], batch_norm=False,last_batchnorm=False):
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
        self.drop_out = drop_out
        self.last_activation = last_activation
        self.return_extra_index = return_extra_index
        self.batch_norm = batch_norm
        self.final_bn = None
        if last_batchnorm:
            self.final_bn = nn.BatchNorm1d(num_neurons[-1],affine=False)

    def forward(self, x):
        num_layer = self.num_layer
        outs_extra = []
        for i in range(num_layer):
            layer_name = "lin{}".format(i+1)
            layer = self.__getattr__(layer_name)
            x = layer(x)

            if self.batch_norm:
                bn_name = "bn{}".format(i+1)
                bn = self.__getattr__(bn_name)
                x = bn(x)

            if i < num_layer - 1:
                if self.drop_out >= 0:
                    x = F.dropout(x, p=self.drop_out, training=self.training)
                x = F.relu(x, inplace=True)

            if (i+1) in self.return_extra_index:
                outs_extra.append(x)

        if self.last_activation == "relu":
            x = F.relu(x, inplace=True)
            if self.final_bn: x = self.final_bn(x)
        elif self.last_activation == "sigmoid":
            x = torch.sigmoid(x)
            if self.final_bn: x = self.final_bn(x)
        elif self.last_activation == "exp_norm":
            x = torch.exp(x - x.max(dim=1)[0].unsqueeze(1))
            if self.final_bn: x = self.final_bn(x)
        elif self.last_activation == "tanh":
            x = torch.tanh(x)
            if self.final_bn: x = self.final_bn(x)
        elif self.last_activation == "softmax":
            if self.final_bn: x = self.final_bn(x)
            x = torch.softmax(x, dim=1)
        elif self.last_activation == 'None':
            if self.final_bn: x = self.final_bn(x)
        else:
            assert TypeError

        if len(outs_extra) > 0:
            return [x] + outs_extra
        else:
            return x

class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)
            self.fc = self.contrastive_head

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features
    
    def set_head(self,layer):
        self.contrastive_head = layer
        #self.fc = self.contrastive_head
        
    def get_head(self):
        return self.contrastive_head
    
    def set_backbone(self,model):
        self.backbone = model
        
    def get_backbone(self):
        return self.backbone


class ClusteringModel(nn.Module): # id = ScanClusteringModel
    def __init__(self, backbone, nclusters, m):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = m['nheads']
        self.aug_type = m['aug_type']
        assert(isinstance(self.nheads, int))

        if self.nheads == 0:
            if m['head_type'] == 'linear':
                self.head = nn.Linear(self.backbone_dim, nclusters)
            else:
                self.head = MLP(num_neurons=m['num_neurons'], drop_out=m['drop_out'], last_activation=m['last_activation'], return_extra_index=[], batch_norm=m['batch_norm'], last_batchnorm=m['last_batchnorm'])
        elif self.nheads > 0:
            if m['head_type'] == 'linear':
                self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
            else:
                self.cluster_head = nn.ModuleList([MLP(num_neurons=m['num_neurons'], drop_out=m['drop_out'], last_activation=m['last_activation'], return_extra_index=[], batch_norm=m['batch_norm'], last_batchnorm=m['last_batchnorm']) for _ in range(self.nheads)])


    def forward(self, x, forward_pass = 'default', aug_type = None ):

        #forward_pass = f_args['forward_pass']
        if not aug_type:
            aug_type = self.aug_type

        if forward_pass == 'default':
            features = self.forward_backbone(x,aug_type)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.forward_backbone(x,aug_type)

        elif forward_pass == 'single_default':
            features = self.forward_backbone(x,aug_type)
            out = self.head(features)

        elif forward_pass == 'single_head':
            #features = self.forward_backbone(x,aug_type)
            out = self.head(features)

        elif forward_pass == 'single_eval':
            features = self.forward_backbone(x,aug_type='eval')
            out = self.head(features)

        elif forward_pass == 'eval':
            print('KEY BLOCK ENTERED !!!!!!!!!!!!!!!!')
            features = self.forward_backbone(x,aug_type='eval')
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.forward_backbone(x,aug_type)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out

    def forward_backbone(self,x,aug_type):

        if aug_type == 'multicrop':
            """
                Codes about multi-crop is borrowed from the codes of Dino
                https://github.com/facebookresearch/dino
            """
            if not isinstance(x, list):
                x = [x]
            # the first indices of aug with changing resolution
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)

            start_idx = 0
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
        
        else: 
            peint('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            output = self.backbone(x)


        return output



class TWIST(nn.Module):
    def __init__(self, hid_dim, clusters ,backbone,backbone_feature_dim):
        super(TWIST, self).__init__()
        #if args['backbone'].startswith('resnet'):
        #widen_resnet.__dict__['resnet50'] = torchvision.models.resnet50
        self.backbone = backbone                   # widen_resnet.__dict__[args['backbone']]()
        self.feature_dim = backbone_feature_dim  # self.backbone.fc.weight.shape[1]
 
        
        #self.backbone.fc = nn.Identity()
        #self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        
        self.head = ProjectionHead_twist(hid_dim, feature_dim=self.feature_dim, clusters=clusters)



    def forward(self, x, aug_type='multicrop',forward_pass='default'):

        features = self.forward_backbone(x,aug_type)
        out = self.head(features)
        return out

    def backbone_weights(self):
        return self.backbone.state_dict()

    def forward_backbone(self,x,aug_type='multicrop'):

        if aug_type == 'multicrop':
            """
                Codes about multi-crop is borrowed from the codes of Dino
                https://github.com/facebookresearch/dino
            """
            if not isinstance(x, list):
                x = [x]
            # the first indices of aug with changing resolution
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)

            start_idx = 0
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
        
        else: output = self.backbone(x)


        return output


class ProjectionHead_twist(nn.Module):
    def __init__(self, hid_dim ,feature_dim ,clusters):
        super(ProjectionHead_twist, self).__init__()
        #dim = args['dim']
        #hid_dim = args['hid_dim']
        norm = nn.BatchNorm1d(clusters,affine=False)
        batchnorm = nn.BatchNorm1d        

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hid_dim, bias=True),
            batchnorm(hid_dim),
            nn.ReLU(),
            #nn.Dropout(p=args['drop']),

            nn.Linear(hid_dim, hid_dim, bias=True),
            batchnorm(hid_dim),
            nn.ReLU(),
        )

        last_linear = nn.Linear(hid_dim, clusters, bias=True)
        self.last_linear = last_linear
        self.norm = norm
        self.num_layer = 3

    def reg_gnf(self, grad):
        self.gn_f = grad.abs().mean().item()

    def reg_gnft(self, grad):
        self.gn_ft = grad.abs().mean().item()

    def forward(self, x):
        x = self.projection_head(x)
        f = self.last_linear(x)
        ft = self.norm(f)
        if self.train and x.requires_grad:
            f.register_hook(self.reg_gnf)
            ft.register_hook(self.reg_gnft)
        self.f_column_std = f.std(dim=0, unbiased=False).mean()
        self.f_row_std    = f.std(dim=1, unbiased=False).mean()
        self.ft_column_std = ft.std(dim=0, unbiased=False).mean()
        self.ft_row_std    = ft.std(dim=1, unbiased=False).mean()

        return ft

class MlpHeadModel(nn.Module):
    def __init__(self, backbone, backbone_dim, args):
        super(MlpHeadModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        args['num_neurons'][0] = backbone_dim
        self.nheads = 1
        self.aug_type = args['aug_type']
        #assert(isinstance(self.nheads, int))
        #assert(self.nheads > 0)
        #self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        self.head = MLP(num_neurons=args['num_neurons'], drop_out=args['drop_out'], last_activation=args['last_activation'], return_extra_index=[], batch_norm=args['batch_norm'],last_batchnorm=args['last_batchnorm'])

    def forward(self,x,forward_pass='default',aug_type=None):

        if not aug_type:
            aug_type = self.aug_type

        if forward_pass == 'default':
            features = self.forward_backbone(x,aug_type)
            return self.head(features)

        elif forward_pass == 'backbone':
            return self.forward_backbone(x,aug_type)

        elif forward_pass == 'head':
            return self.head(x)

        elif forward_pass == 'eval':
            features = self.forward_backbone(x,aug_type='eval')
            return self.head(features)

        else: raise ValueError

    def forward_backbone(self,x,aug_type='multicrop'):

        if aug_type == 'multicrop':

            if not isinstance(x, list):
                x = [x]
            # the first indices of aug with changing resolution
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)

            start_idx = 0
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx

        else: output = self.backbone(x)


        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}


def load_backbone_model(model,path,backbone_type):
    
    pretrained = torch.load(path,map_location='cpu')

    if backbone_type == 'lightly_resnet18':
        model.backbone.load_state_dict(pretrained,strict=True)

    elif backbone_type == 'clPcl':
        missing = model['backbone'].load_state_dict(pretrained, strict=False)
        print("missing layers: ",missing)

    elif backbone_type == 'scatnet':
        model.load_state_dict(pretrained,strict=True)

    else: raise ValueError
