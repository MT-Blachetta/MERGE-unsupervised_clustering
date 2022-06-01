from models import construct_spice_model, load_spice_model
from functionality import collate_custom
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn import cluster
import sklearn
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as transforms
from scatnet import ScatSimCLR
#from utils import compute_scan_dataset, compute_default_dataset

def main():

    prefix = 'spice-scatnet_5head_batchnorm'
    model_type = 'spice_batchnormMLP'
    savefile = "/home/blachm86/SPICE-main/results/stl10/scatnet_self_5head_batchnorm/checkpoint_select.pth.tar" # put here the path to the model file 
    device = 'cuda:0'

    p = {

    'val_split': 'both',
    'dataset_type': 'scan',
    'transformation_kwargs': {'crop_size': 96, 'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'train_db_name': 'stl-10',
    'to_neighbors_dataset': True, 
    'topk_neighbors_val_path': 'RESULTS/stl-10/topk/both_scatnet_topk-val-neighbors.npy',
    'num_workers': 8,
    'batch_size': 256

    }

    m_args = {'model_type': model_type, 
            'num_neurons': [128,128,10], 
            'last_activation': 'softmax', 
            'batch_norm': True,
            'last_batchnorm': False,
            'J': 2,
            'L': 16,
            'input_size': [96,96,3],
            'res_blocks': 30,
            'out_dim': 128
            }

 
    backbone = ScatSimCLR(J=m_args['J'], L=m_args['L'], input_size=tuple(m_args['input_size']), res_blocks=m_args['res_blocks'],out_dim=m_args['out_dim'])

    # get the model from file to the code
    model = construct_spice_model(m_args,backbone)
    load_spice_model(model,savefile,model_type)

    from dataset.scan_dataset import STL10_trainNtest, STL10_eval


    val_transformations = transforms.Compose([
                                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(**p['transformation_kwargs']['normalize'])])
        
    val_split = p['val_split']
    dataset_type = p['dataset_type']

    if dataset_type == 'scan':

        if p['train_db_name'] == 'cifar-10':
            from dataset.scan_dataset import CIFAR10
            #dataset = CIFAR10(train=True, transform=train_transformation, download=True)
            eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'cifar-20':
            from dataset.scan_dataset import CIFAR20
            #dataset = CIFAR20(train=True, transform=train_transformation, download=True)
            eval_dataset = CIFAR20(train=False, transform=val_transformations, download=True)

        elif p['train_db_name'] == 'stl-10':
            if val_split == 'train':
                from dataset.scan_dataset import STL10
                eval_dataset = STL10(split='train', transform=val_transformations, download=False)
            elif val_split == 'test':
                eval_dataset = STL10(split='test', transform=val_transformations, download=False)
            elif val_split == 'both':
                eval_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)
            else: raise ValueError('Invalid stl10 split')

                #print('eval_dataset:len: ',len(eval_dataset))
                #eval_dataset = STL10(split='train',transform=val_transformations,download=False)

        else:
            raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
            

        if p['to_neighbors_dataset']: # Dataset returns an image and one of its nearest neighbors.
            from dataset.scan_dataset import NeighborsDataset
            #print(p['topk_neighbors_train_path'])
            indices = np.load(p['topk_neighbors_val_path'])
            #print(indices.shape)
            eval_dataset = NeighborsDataset(eval_dataset, indices) # , p['num_neighbors'])

        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=False, shuffle=False)
                
                

    else:
        #dataset = torchvision.datasets.STL10('/space/blachetta/data', split=split,transform=train_transformation, download=True)
        eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)

        val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=p['num_workers'],
                                                    batch_size=p['batch_size'], pin_memory=True,
                                                    drop_last=False, shuffle=False)


    metric_data = Analysator(device,model,val_dataloader)
    metric_data.compute_kNN_statistics(100)
    metric_data.compute_real_consistency(0.5)
    metric_data.return_statistic_summary()

    pickle.dump(metric_data, open('SPICE_EVAL/'+prefix+".pck",'wb'))

#---------------------------------------------------------------------------------------------

class Analysator():
    def __init__(self,device,model,dataloader,class_names=['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']):
         
        model.eval()
        predictions = []
        labels = []
        features = []
        soft_labels = []
        confidences = []
        #self.run_parameters = run_parameters

        model.to(device)


        with torch.no_grad():      
            for batch in dataloader:
                if isinstance(batch,dict):
                    image = batch['image']
                    label = batch['target']
                else:
                    image = batch[0]
                    label = batch[1]

                image = image.to(device,non_blocking=True)
                fea = model(image,forward_type="features")
                features.append(fea)
                preds = model(fea,forward_type="head")
                soft_labels.append(preds)
                max_confidence, prediction = torch.max(preds,dim=1) 
                predictions.append(prediction)
                confidences.append(max_confidence)
                labels.append(label)

        self.feature_tensor = torch.cat(features)
        self.softlabel_tensor = torch.cat(soft_labels)
        self.prediction_tensor = torch.cat(predictions)
        self.label_tensor = torch.cat(labels)
        self.confidence_tensor = torch.cat(confidences)

        self.classes = [ class_names[l.item()] for l in self.label_tensor  ]

        self.dataset_size = self.label_tensor.shape[0]

        self.feature_tensor = torch.nn.functional.normalize(self.feature_tensor, dim = 1)
        self.similarity_matrix = torch.einsum('nd,cd->nc', [self.feature_tensor.cpu(), self.feature_tensor.cpu()])

        y_train = self.label_tensor.detach().cpu().numpy()
        pred = self.prediction_tensor.detach().cpu().numpy()
        max_label = max(y_train)
        assert(max_label==9)

        self.C = get_cost_matrix(pred, y_train, max_label+1)
        ri, ci = assign_classes_hungarian(self.C)

        self.cluster_to_class = torch.Tensor(ci)
        self.correct_samples = self.cluster_to_class[self.prediction_tensor] == self.label_tensor
        self.bad_samples = self.correct_samples == False
        #self.kNN_cosine_similarities = None
        self.kNN_indices = None
        self.kNN_labels = None
        self.kNN_consistent = None
        self.kNN_confidences = None
        self.proximity = None # mean distance of the top nearest neigh
        self.local_consistency = None
        self.criterion_consistent = None
        self.summary = None
        self.knn = 0

        #class_names = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']


    #def compute_correct_samples_mask(self):
        
    #    self.correct_samples = self.cluster_to_class[self.prediction_tensor] == self.label_tensor

    def compute_kNN_statistics(self,knn):
        self.knn = knn
        scores_k, idx_k = self.similarity_matrix.topk(k=knn, dim=1)
        self.proximity = torch.mean(scores_k,dim=1)
        self.kNN_indices = idx_k
        labels_topk = torch.zeros_like(idx_k)
        confidence_topk = torch.zeros_like(idx_k,dtype=torch.float)
        for s in range(knn):
            labels_topk[:, s] = self.prediction_tensor[idx_k[:, s]]
            confidence_topk[:, s] = self.confidence_tensor[idx_k[:, s]]
        
        self.kNN_consistent = labels_topk[:, 0:1] == labels_topk # <boolean mask>
        self.local_consistency = self.kNN_consistent.sum(dim=1)/knn
        self.kNN_labels = labels_topk
        self.kNN_confidences = confidence_topk
        # condition = self.kNN_confidences > 0.5
        
        
    def compute_real_consistency(self, criterion):

        self.criterion_consistent = []
        for i in range(self.dataset_size):
            confids = self.kNN_confidences[i][self.kNN_consistent[i]] # +logical_index > +true for index of consistent label; +size=knn > +indexes topk instances
            real = confids > criterion
            self.criterion_consistent.append(sum(real)/self.knn)

        self.criterion_consistent = torch.Tensor(self.criterion_consistent)

    def return_statistic_summary(self):
        statistic = pd.Series()

        #statistic['Loss'] = best_loss
        statistic['Accuracy'] = self.get_accuracy()
        y_train = self.label_tensor.detach().cpu().numpy()
        pred = self.prediction_tensor.detach().cpu().numpy()

        y_pred = pred
        y_true = y_train

        ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)
        v_measure = sklearn.metrics.v_measure_score(y_true, y_pred)
        ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
        fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)

        statistic['Adjusted_Mutual_Information'] = ami
        statistic['Adjusted_Random_index'] = ari
        statistic['V_measure'] = v_measure
        statistic['fowlkes_mallows'] = fm

        cluster_metric = self.categorical_from_selection(self.prediction_tensor)

        statistic['cluster_size_min'] = min([ sum(self.C[ci,:]) for ci in range(10) ])

        #cluster_metric['min']
        statistic['cluster_entropy'] = cluster_metric['entropy']

        statistic['consistency_ratio'] = self.num_of_consistents(upper=0.5,lower=1.0,real_consistent=True)/self.dataset_size
        statistic['confidence_ratio'] = self.num_of_confidents(0.95)/self.dataset_size

        statistic['correct_mean_confidence'] = self.mean_from_selection(self.correct_samples.cpu().numpy(),self.confidence_tensor.cpu().numpy())
        statistic['bad_mean_confidence'] = self.mean_from_selection(self.bad_samples.cpu().numpy(),self.confidence_tensor.cpu().numpy())
        statistic['correct_mean_consistency'] = self.mean_from_selection(self.correct_samples.cpu().numpy(),self.criterion_consistent.cpu().numpy())
        statistic['bad_mean_consistency'] = self.mean_from_selection(self.bad_samples.cpu().numpy(),self.criterion_consistent.cpu().numpy())

        self.summary = statistic

        return statistic

        
        
    def get_accuracy(self):

        return to_value(sum(self.correct_samples)/len(self.correct_samples))
        

    def get_meanConfidence_of_consistents(self):
        
        means = []
        for i in range(self.dataset_size):
            confids = self.kNN_confidences[i][self.kNN_consistent[i]] # +logical_index > +true for index of consistent label; +size=knn > +indexes topk instances
            mean_confidence = sum(confids)/len(confids) # OK
            means.append(mean_confidence)

        return torch.Tensor(means)
        

    def get_meanConsistency_of_confidents(self,criterion):

        #means = []
        conf_mask = self.confidence_tensor > criterion
        if sum(conf_mask) == 0: return nan
        consistents = self.local_consistency[conf_mask] # generic for selection masks
        return to_value(sum(consistents)/len(consistents))
        
    def num_of_confidents(self,upper,lower=1.0):

        upmask = self.confidence_tensor > upper
        lowmask = self.confidence_tensor <= lower

        conf_mask = upmask*lowmask

        return to_value(sum(conf_mask))

    def num_of_consistents(self,upper,lower=1.0,real_consistent=False): # divide through dataset_size to get rate/ratio

        if real_consistent:

            upmask = self.criterion_consistent > upper
            lowmask = self.criterion_consistent <= lower

            consistent_mask = upmask*lowmask            

        else:

            upmask = self.local_consistency > upper
            lowmask = self.local_consistency <= lower

            consistent_mask = upmask*lowmask

        return to_value(sum(consistent_mask))        
        
    def num_reliable_criterion(self,consistent_ratio,confidence_ratio,real_consistent=False):
       
        confidence_mask = self.select_confident(confidence_ratio)
        consistent_mask = self.select_local_consistent(consistent_ratio,real_consistent=real_consistent)

        mask = confidence_mask*consistent_mask

        return to_value(sum(mask))
        

    def select_reliable_criterion(self,consistent_ratio,confidence_ratio,real_consistent=False):

        confidence_mask = self.select_confident(confidence_ratio)
        consistent_mask = self.select_local_consistent(consistent_ratio,real_consistent=real_consistent)

        mask = confidence_mask*consistent_mask

        return mask.type(torch.bool)        
        
     

    def select_confident(self,upper,lower=1.0):

        upmask = self.confidence_tensor > upper
        lowmask = self.confidence_tensor <= lower

        conf_mask = upmask*lowmask

        return conf_mask.type(torch.bool) # in dataset size
        

    def select_local_consistent(self,upper,lower=1.0,real_consistent=False):

        if real_consistent:

            upmask = self.criterion_consistent > upper
            lowmask = self.criterion_consistent <= lower

            consistent_mask = upmask*lowmask

        else:

            upmask = self.local_consistency > upper
            lowmask = self.local_consistency <= lower

            consistent_mask = upmask*lowmask

        return consistent_mask.type(torch.bool) # in dataset size        
                
       
        
    def get_accuracy_from_selection(self,selection_mask):

        correct_subset = self.correct_samples[selection_mask]

        return to_value(sum(correct_subset)/len(correct_subset))
        
        
    def ratio_from_selection(self,selection_mask,boolean_tensor,relative=True):
        
        indicators = boolean_tensor[selection_mask]
        if len(indicators) == 0: return 0 
        
        if relative:
            return to_value(sum(indicators)/len(indicators))
        else:
            return to_value(sum(indicators))

    def mean_std_from_selection(self,selection_mask,value_tensor):

        sub_features = value_tensor[selection_mask]
        if len(sub_features) == 0: return nan, nan 
        std, mean = torch.std_mean(sub_features,unbiased=False)

        return to_value(mean), to_value(std)
        

    def categorical_from_selection(self,category_mapping,selection_mask=None,values_to_count=None,mode='ratio+entropy',return_type='pandas'):

        if selection_mask is None:
            selection_mask = torch.full([self.dataset_size],True)

        if isinstance(category_mapping,torch.Tensor):
            bins = torch.unique(category_mapping)
        else: bins = set(category_mapping)

        category_values = category_mapping[selection_mask]

        if values_to_count is not None: measurements = values_to_count[selection_mask]
        else: measurements = None

        total = len(category_values)
        if total == 0:
            if return_type == 'list': return [str(c.item()) for c in bins], [0 for _ in bins]
            if return_type == 'pandas': 
                if 'entropy' in mode: columnlist = ['entropy', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                else: columnlist = [ 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                result = pd.Series([0 for _ in bins],index=[str(c.item()) for c in bins])
                for n in columnlist: result[n] = nan
                return result

        category_names = []
        amounts = []

        for c in bins:

            category_names.append(str(c.item()))
            counting_mask = category_values == c
            end_value = self.count_set(counting_mask,measurements,mode)
            amounts.append(end_value)

        if return_type == 'list': 
            return category_names, amounts

        elif return_type == 'pandas':
            result = pd.Series(amounts,index=category_names)
            statistics = result.describe()
            statistics['sum'] = result.sum()
            #result.update(statistics)
            

            if 'entropy' in mode:
                columnlist = ['entropy', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                entropy = self.entropy_from_ratios(torch.Tensor(amounts))
                statistics['entropy'] = entropy
                for name in columnlist:
                    result[name] = statistics[name]           

            else:                
                columnlist = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
                for name in columnlist:
                    result[name] = statistics[name]

        
        return result


    def mean_from_selection(self,selection_mask,scalar_tensor):
        
        if selection_mask is None:
            selection_mask = torch.full([self.dataset_size],True)

        subset_values = scalar_tensor[selection_mask]
        subset_size = len(subset_values)

        if subset_size == 0: return 0

        return sum(subset_values)/len(subset_values)

        



    def scalar_statistics_from_selection(self,selection_mask,scalar_tensor,parameters):

        # init ------------------------------------

        if selection_mask is None:
            selection_mask = torch.full([self.dataset_size],True)

        if parameters['secondary']:
            val_range = parameters['range_2']
            interval = parameters['interval_2']
        else:
            val_range = parameters['range']
            interval = parameters['interval']


        returntype = parameters['return_type']
        
        if 'count_measure' in parameters.keys(): 
            values_to_count = parameters['count_measure']
            count_mode = parameters['count_mode']
            measurements = values_to_count[selection_mask]
        else: measurements = None

        subset_values = scalar_tensor[selection_mask]
        subset_size = len(subset_values)

        # initialize index or column names
        names = []
        values = []
        iv = val_range[0]

        while iv <= val_range[1]:            
            names.append(str(iv))
            iv += interval

        # zero case ---------------------------------

        if subset_size == 0: 
            #subset_values = torch.Tensor([0])
            values = [0 for _ in names]
            #numbers = [0 for _ in names]
            frequencies = pd.Series(values,index=names)
            columnlist = ['count' ,'sum', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            for n in columnlist: frequencies[n] = nan
            return frequencies
                
        
        #return_info = {}
        #names = [str(range[0])]
        #values = [0]
        numbers = []
        
        #high = max(scalar_tensor)
        iv = val_range[0]-interval
        

        while (iv+interval) <= val_range[1]:
            
            #names.append(str(iv+interval))
            cmask = self.double_condition_mask(iv,iv+interval,subset_values)
            counted_value = self.count_set(cmask,measurements,count_mode)
            values.append(counted_value)
            numbers.append(self.count_double_condition(iv,iv+interval,subset_values))
            iv += interval

        if returntype == 'dict':
            ps = pd.Series(subset_values.detach().cpu().numpy())
            description = ps.describe()
            resultdict = {'real_values': subset_values, 'intervals':names, 'ratios':values, 'numbers':numbers}
            resultdict.update(description)
            
            return resultdict

        #g = nan is set is zero
            
        elif returntype == 'pandas':
            ps = pd.Series(subset_values.detach().cpu().numpy())
            description = ps.describe()
            frequencies = pd.Series(values,index=names)
            columnlist = ['count' ,'sum', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            #names.extend(columnlist)  # CAN BE USED LATER !
            description['sum'] = ps.sum()
            for n in columnlist:
                frequencies[n] = description[n]

            return frequencies

        else:
            return names, values 



    def two_scalarSet_statistics_from_selection(self,scalar_features,second_feature,parameters,dataset_mask=None):

        result_frame = pd.DataFrame()

        if dataset_mask is None:
            selected_features = scalar_features
            subfeature_values = second_feature
            subset_size = len(self.dataset_size)

        else:
            selected_features = scalar_features[dataset_mask]
            subfeature_values = second_feature[dataset_mask]
            subset_size = len(subfeature_values)

        val_range = parameters['range']
        interval = parameters['interval']
        #returntype = parameters['return_type'] # pandas or dictionary

        
        if len(subset_size) == 0: 
            selected_features = torch.Tensor([0])
            subfeature_values = torch.Tensor([0])

        #return_info = {}
        names = []
        #values = [0]
        #numbers = [0]

        #high = max(scalar_tensor)
        iv = val_range[0]
        
        while iv <= val_range[1]:            
            names.append(str(iv))
            iv += interval

        iv = val_range[0] - interval
        parameters['secondary'] = True

        #first_set = selected_features == iv
        #first_row = self.scalar_statistics_from_selection(first_set,subfeature_values,parameters)
        #result_frame = pd.DataFrame(dict(first_row),columns=first_row.index,index=[names[0]])
        ni = 0

        while iv+interval <= val_range[1]:

            next_set = self.double_condition_mask(iv,iv+interval,selected_features)
            next_row = self.scalar_statistics_from_selection(next_set,subfeature_values,parameters)
            added = pd.DataFrame(dict(next_row),columns=next_row.index,index=[names[ni]])
            result_frame = pd.concat([result_frame,added])
            iv += interval
            ni += 1
    

        return result_frame


    def two_categorical_statistics_from_selection(self,category_features,second_feature,parameters,dataset_mask=None):

        result_frame = pd.DataFrame()

        if dataset_mask is None:
            selected_features = category_features
            subfeature_values = second_feature
            subset_size = len(self.dataset_size)

        else:
            selected_features = category_features[dataset_mask]
            subfeature_values = second_feature[dataset_mask]
            subset_size = len(subfeature_values)

        
        #returntype = parameters['return_type'] # pandas or dictionary
       
        if len(subset_size) == 0: 
            selected_features = torch.Tensor([0])
            subfeature_values = torch.Tensor([0])

        bins = torch.unique(category_features)

        second_type = parameters['secondary_type']

        if second_type == 'categorical':
        
            vtc = None
            mode = parameters['mode'] 
            rtype = parameters['return_type']
            
            if 'count_measure' in parameters.keys():
                valuesToCount = parameters['count_measure']

                # values_to_count is in DATASET-SIZE, so it must be adapted
                if dataset_mask is not None: vtc = valuesToCount[dataset_mask]
                else: vtc = valuesToCount

            for v in bins:
                subcategory_mask = selected_features == v
                rowSeries = self.categorical_from_selection(subfeature_values,subcategory_mask,values_to_count=vtc,mode=mode,return_type=rtype)
                result_frame = pd.concat([result_frame,rowSeries])

        else:
            #valuesToCount = parameters['']
            if 'count_measure' in parameters.keys(): 
                values_to_count = parameters['count_measure']
                if dataset_mask is not None: parameters['count_measure'] = values_to_count[dataset_mask]
            
            for v in bins:
                subcategory_mask = selected_features == v
                rowSeries = self.scalar_statistics_from_selection(subcategory_mask,subfeature_values,parameters)
                result_frame = pd.concat([result_frame,rowSeries])

        return result_frame


    def entropy_from_ratios(self,fractions):
        fractions += 0.00001
        return to_value(-sum(fractions*torch.log(fractions)))


    def count_double_condition(self,low,high,value_set):

        cond1 = value_set <= high 
        cond2 = value_set > low

        return to_value(sum(cond1*cond2))


    def double_condition_mask(self,low,high,value_set):

        cond1 = value_set <= high 
        cond2 = value_set > low

        mask = cond1*cond2

        return mask.type(torch.bool)


    def count_set(self,set_mask,measurements,count_mode='mean'):

        if sum(set_mask) == 0: return 0

        if measurements is None:
            if 'ratio' in count_mode:
                return to_value(sum(set_mask)/len(set_mask))
            else: return to_value(sum(set_mask))

        else: 

            if 'mean' in count_mode:
                feature_values = measurements[set_mask]
                return to_value(sum(feature_values)/len(feature_values))
            else: return to_value(sum(feature_values)) 

def to_value(v):
    if isinstance(v,torch.Tensor):
        v = v.item()        
    return v

def get_cost_matrix(y_pred, y, nc=1000): # C[ground-truth_classes,cluster_labels] counts all instances with a given ground-truth and cluster_label
    C = np.zeros((nc, y.max() + 1))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
    return C 

def assign_classes_hungarian(C): # rows are (num. of) clusters and columns (num. of) ground-truth classes
    row_ind, col_ind = linear_sum_assignment(C, maximize=True) # assume 1200 rows(clusters) and 1000 cols(classes)
    ri, ci = np.arange(C.shape[0]), np.zeros(C.shape[0]) # ri contains all CLUSTER/CLASS indexes as integer from 0 / ci the assigned class/cluster --> num_classes
    ci[row_ind] = col_ind # assignment of the col_ind[column nr. = CLASS_ID] to the [row nr. = cluster_ID/index]

    # ri =: cluster
    # ci =: class of cluster corresponded by index

    # for overclustering, rest is assigned to best matching class
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False # True = alle cluster die nicht durch [linear_sum_assignment] einer Klasse zugeordnet wurden
    ci[mask] = C[mask, :].argmax(1) # Für weitere Cluster über die Anzahl Klassen hinaus, ordne die Klasse mit der größten Häufigkeit zu 
    return ri.astype(int), ci.astype(int) # at each position one assignment: ri[x] = index of cluster <--> ci[x] = classID assigned to cluster

if __name__ == "__main__":
    main()