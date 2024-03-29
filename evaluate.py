from cmath import nan
import torch
import numpy as np
from sklearn.utils import shuffle
from sklearn import cluster
import sklearn
from sklearn.decomposition import IncrementalPCA
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment
import pandas as pd



#@author: Michael Blachetta
def topk_consistency(features,predictions,num_neighbors):
    
    features = torch.nn.functional.normalize(features, dim = 1)
    similarity_matrix = torch.einsum('nd,cd->nc', [features.cpu(), features.cpu()])
    scores_k, idx_k = similarity_matrix.topk(k=num_neighbors, dim=1)
    labels_samples = torch.zeros_like(idx_k)

    for s in range(num_neighbors):
        labels_samples[:, s] = predictions[idx_k[:, s]]
    
    true_matrix = labels_samples[:, 0:1] == labels_samples
    num_consistent = true_matrix.sum(dim=1)

    return num_consistent/num_neighbors


#@author: Michael Blachetta
def cluster_size_entropy(costmatrix):

    absolute = costmatrix.sum(axis=1)
    relative = (absolute/sum(absolute)) + 0.00001
    entropy = - sum(relative*np.log(relative))
    
    return entropy


#@author: Michael Blachetta
def confidence_statistic(softmatrix):
    max_confidences, _ = torch.max(softmatrix,dim=1)
    num_confident_samples = len(torch.where(max_confidences > 0.95)[0])
    confidence_ratio = num_confident_samples/len(max_confidences)
    confidence_std, confidence_mean = torch.std_mean(max_confidences, unbiased=True)

    return confidence_mean, confidence_std, confidence_ratio
    


def batches(l, n):
    for i in range(0, len(l), n): # step_size = n, [i] is a multiple of n
        yield l[i:i + n] # teilt das array [l] in batch sub_sequences der Länge n auf


#@composed and adapted: Michael Blachetta
def get_cost_matrix(y_pred, y, nc=1000): # C[ground-truth_classes,cluster_labels] counts all instances with a given ground-truth and cluster_label
    C = np.zeros((nc, y.max() + 1))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
    return C 


#@composed and adapted: Michael Blachetta
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


#@composed and adapted: Michael Blachetta
def assign_classes_majority(C):
    col_ind = C.argmax(1) # assign class with the highest occurence to the cluster (row)
    row_ind = np.arange(C.shape[0]) # clusterID at position in both arrays (col_ind and row_ind)

    # best matching class for every cluster
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False

    return row_ind.astype(int), col_ind.astype(int)


#@composed and adapted: Michael Blachetta
def accuracy_from_assignment(C, row_ind, col_ind, set_size=None):
    if set_size is None:
        set_size = C.sum()
    cnt = C[row_ind, col_ind].sum() # sum of all correctly (class)-assigned instances that contributes to the Cluster's ClassID decision
    # (that caused the decision)
    return cnt / set_size # If all clusters would have only instaces of one unique class, this value becomes = 1


#@composed and adapted: Michael Blachetta
def get_best_clusters(C, k=3, formatation=False):
    Cpart = C / (C.sum(axis=1, keepdims=True) + 1e-5) # relative Häufigkeit für jedes Cluster label
    Cpart[C.sum(axis=1) < 10, :] = 0 # Schwellwert für die Mindestanzahl Instanzen mit ground-truth_class
    # setzt bestimmte relative Häufigkeiten auf 0 (aus der Bewertung entfernt)
    # print('as', np.argsort(Cpart, axis=None)[::-1])
    
    # np.argsort(Cpart, axis=None)[::-1] # flattened indices in umgekehrt_absteigender Abfolge (sonst aufsteigender Reihenfolge)
    # Cpart.shape = (1000,1000)
    ind = np.unravel_index(np.argsort(Cpart, axis=None)[::-1], Cpart.shape)[0]  # first-dimension indices in C of good clusters (highest single frequency correlation)
    _, idx = np.unique(ind, return_index=True) # index of the first occurence of the unique element in $[ind]
    # idx = 1000 aus einer Million indices (höchst-erst-bestes aus jeder ground-truth), keine Duplikate
    cluster_idx = ind[np.sort(idx)]  # unique indices of good clusters (von groß nach klein)
    # nimmt den ersten Wert eines auftauchenden classIndex value von [ind] und notiert sich nur die Indexposition in [ind] dabei
    # die Werte werden von Beginn bis Ende in der Reihenfolge von [ind] ausgewählt; somit ist der kleinste Wert von idx auch 
    # der erste Wert von [ind], weitere Werte mit dem gleichen classIndex werden übersprungen und der zweite Werte ist somit der
    # nächsthöchste classIndex von [ind], somit hat man die besten classID's in absteigender Reihenfolge    
    accs = Cpart.max(axis=1)[cluster_idx] # die accuracies (höchste Wahrscheinlichkeit von Cpart) der besten classes/cluster (als ID)
    good_clusters = cluster_idx[:k] # selects the k best clusters
    best_acc = Cpart[good_clusters].max(axis=1)
    best_class = Cpart[good_clusters].argmax(axis=1)
    #print('Best clusters accuracy: {}'.format(best_acc))
    #print('Best clusters classes: {}'.format(best_class))
    """
    if formatation:
        outstring = ''
        for i in range(k):
            outstring += str(i)
            outstring += ' ,'
            outstring += str(good_clusters[i])
            outstring += ','
            outstring += str(best_class[i])
            outstring += ','
            outstring += str(best_acc[i])        
            outstring += '\n'
            
        print(outstring)
    """
  
    return {'best_clusters': good_clusters, 'classes': best_class, 'accuracies': best_acc}


#@composed and adapted: Michael Blachetta
def train_pca(X_train,n_comp):
    bs = max(4096, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs,n_components=n_comp)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs + 1)):
        transformer = transformer.partial_fit(batch)
        # break
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer


#@composed and adapted: Michael Blachetta
def transform_pca(X, transformer):
    n = max(4096, X.shape[1] * 2)
    n_comp = transformer.components_.shape[0]
    X_ = np.zeros((X.shape[0],n_comp))
    for i in trange(0, len(X), n):
        X_[i:i + n] = transformer.transform(X[i:i + n])
        # break
    return X_


#@author: Michael Blachetta
@torch.no_grad()
def evaluate_singleHead(device,model,dataloader,forwarding='head',formatation=False):

    model.eval()
    predictions = []
    labels = []
    features = []
    soft_labels = []

    model.to(device)
    #type_test = next(iter(dataloader))
    #isinstance

    with torch.no_grad():      
        for batch in dataloader:
            if isinstance(batch,dict):
                image = batch['image']
                label = batch['target']
            else:
                image = batch[0]
                label = batch[1]

            image = image.to(device,non_blocking=True)
            fea = model(image,forward_pass='features')
            features.append(fea)
            predic = model(fea,forward_pass=forwarding)
            soft_labels.append(predic)
            predictions.append(torch.argmax(predic, dim=1))
            labels.append(label)

    feature_tensor = torch.cat(features)
    softlabel_tensor = torch.cat(soft_labels)
    prediction_tensor = torch.cat(predictions)
    label_tensor = torch.cat(labels)

    consistency_values = topk_consistency(feature_tensor,prediction_tensor,100)
    consistency_ratio = len(torch.where(consistency_values > 0.5)[0])/len(consistency_values)
    print('consistency_ratio = ',consistency_ratio)
    #c_std, c_mean =  torch.std_mean(consistency_values, unbiased=False)
    
    y_train = label_tensor.detach().cpu().numpy()
    pred = prediction_tensor.detach().cpu().numpy()
    max_label = max(y_train)
    assert(max_label==9)

    C_train = get_cost_matrix(pred, y_train, max_label+1)

    cluster_entropy = cluster_size_entropy(C_train)
    conf_mean, conf_std, conf_rate = confidence_statistic(softlabel_tensor)
    print('confidence_rate = ',conf_rate)

    result_dict = {'cluster_size_entropy': cluster_entropy, 'confidence_ratio': conf_rate , 'mean_confidence': conf_mean.item(), 'std_confidence': conf_std.item(), 'consistency_ratio': consistency_ratio}

    message = 'val'
    y_pred = pred
    y_true = y_train
    train_lin_assignment = assign_classes_hungarian(C_train)
    #train_maj_assignment = assign_classes_majority(C_train)

    acc_tr_lin = accuracy_from_assignment(C_train, *train_lin_assignment)
    #get_best_clusters(C_train,k=10,formatation=formatation)--------------------

    #acc_tr_maj = accuracy_from_assignment(C_train, *train_maj_assignment)

    #result_dict = get_best_clusters(C_train,k=10,formatation=formatation)


    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)
    v_measure = sklearn.metrics.v_measure_score(y_true, y_pred)
    ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
    fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)

    #headline = 'method,ACC,ARI,AMI,FowlkesMallow,'
    #print('\ncluster performance:\n')
    #print(eval_name+'  ,'+str(acc_tr_lin)+', '+str(ari)+', '+str(v_measure)+', '+str(ami)+', '+str(fm))

    result_dict['Accuracy'] = acc_tr_lin
    result_dict['Adjusted_Random_Index'] = ari
    result_dict['V_measure'] = v_measure
    result_dict['fowlkes_mallows'] = fm
    result_dict['Adjusted_Mutual_Information'] = ami

    print("\n{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}\tACC {:.5e}".format(message, ari, v_measure, ami, fm, acc_tr_lin))

    return result_dict


#@author: Michael Blachetta
def evaluate_prediction(y_true,y_pred,formatation=False):

    max_label = max(y_true)
    assert(max_label==9)
    #print(y_true)
    C_train = get_cost_matrix(y_pred, y_true, max_label+1)

    #message = 'val'
    #y_pred = pred
    #y_true = y_train
    train_lin_assignment = assign_classes_hungarian(C_train)
    train_maj_assignment = assign_classes_majority(C_train)

    acc_tr_lin = accuracy_from_assignment(C_train, *train_lin_assignment)
    #acc_tr_maj = accuracy_from_assignment(C_train, *train_maj_assignment)

    result_dict = get_best_clusters(C_train,k=10,formatation=formatation)


    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)
    v_measure = sklearn.metrics.v_measure_score(y_true, y_pred)
    ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
    fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)

    #headline = 'method,ACC,ARI,AMI,FowlkesMallow,'
    #print('\ncluster performance:\n')
    #print(eval_name+'  ,'+str(acc_tr_lin)+', '+str(ari)+', '+str(v_measure)+', '+str(ami)+', '+str(fm))

    result_dict['ACC'] = acc_tr_lin
    result_dict['ARI'] = ari
    result_dict['V_measure'] = v_measure
    result_dict['fowlkes_mallows'] = fm
    result_dict['AMI'] = ami

    #print("\n{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}\tACC {:.5e}".format(message, ari, v_measure, ami, fm, acc_tr_lin))

    return result_dict


#@author: Michael Blachetta
def evaluate_headlist(device,model,dataloader,formatation=False):

    predictions = [ [] for _ in range(model.nheads) ]
    label_list = []
    #labels = [ [] for _ in range(model.nheads)]
    model.to(device)
    model.eval()

    with torch.no_grad(): 
        for batch in dataloader:
            if isinstance(batch,dict):
                image = batch['image']
                labels = batch['target']
            else:
                image = batch[0]
                labels = batch[1]

            label_list.append(labels)
            image = image.to(device,non_blocking=True)
            predlist = model(image,forward_pass='eval')
            for k in range(len(predlist)):
                predictions[k].append(predlist[k])

    targets = torch.cat(label_list)
    targets = targets.detach().cpu().numpy()
    #print('targets.shape: ', targets.shape)
    headlist = [torch.cat(pred) for pred in predictions]
    head_labels = [torch.argmax(softlabel,dim=1) for softlabel in headlist]
    #print('len = ',len(headlist))
    #print('predictions.shape: ',headlist[0].shape)

    accuracies = []
    dicts = []
    for h in head_labels:
        rdict = evaluate_prediction(targets,h.detach().cpu().numpy())
        accuracies.append(rdict['ACC'])
        #print(rdict['ACC'])
        dicts.append(rdict)

    best_head = np.argmax(np.array(accuracies))
    best_accuracy = max(accuracies)

    #result_dict = dicts[best_head]
    #result_dict['head_id'] = best_head

    #acc_tr_lin = result_dict['ACC'] 
    #ari = result_dict['ARI'] 
    #v_measure = result_dict['V_measure']
    #fm = result_dict['fowlkes_mallows']
    #ami = result_dict['AMI']

    #message = 'validation'

    print('best accuracy: ', best_accuracy,'  on head ',best_head)
    #print('best head is ',best_head)
    #print("\n{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}\tACC {:.5e}".format(message, ari, v_measure, ami, fm, acc_tr_lin))

    return dicts



#@author: Michael Blachetta
def to_value(v):
    if isinstance(v,torch.Tensor):
        v = v.item()        
    return v
        

#@author: Michael Blachetta
class Analysator():
    def __init__(self,device,model,dataloader,forwarding='head',model_type='cluster_head',labeled=True,class_names=['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']):
         
        self.feature_tensor = None
        self.softlabel_tensor = None
        self.prediction_tensor = None
        self.label_tensor = None
        self.confidence_tensor = None
        self.confidences = None
        self.classes = None
        self.class_names = None
        self.dataset_size = 0
        self.feature_tensor = None
        self.C = None
        self.cluster_to_class = None
        self.cluster_names = None
        self.class_of_prediction = None
        self.clusters = None

        self.correct_samples = None
        self.bad_samples = None

        self.kNN_indices = None
        self.kNN_labels = None
        self.kNN_consistent = None
        self.kNN_confidences = None
        self.proximity = None # mean distance of the top nearest neighbors
        self.local_consistency = None
        self.criterion_consistent = None
        self.summary = None
        self.knn = 100

        if labeled:
            self.initialize_labeled(device,model,dataloader,forwarding,model_type,class_names)
        else:
            self.initialize_unlabeled(device,model,dataloader,forwarding,model_type)


    def initialize_labeled(self,device,model,dataloader,forwarding='head',model_type='cluster_head',class_names=['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']):
        
        model.eval()
        predictions = []
        labels = []
        features = []
        soft_labels = []
        confidences = []
       

        model = model.to(device)


        with torch.no_grad():      
            for batch in dataloader:
                if isinstance(batch,dict):
                    image = batch['image']
                    label = batch['target']
                else:
                    image = batch[0]
                    label = batch[1]

                if model_type == 'contrastive_clustering':
                    image = image.to(device,non_blocking=True)
                    feats, _,preds, _ = model(image,image)
    
                elif model_type == 'fixmatch_model':
                    image = image.to(device,non_blocking=True)
                    feats = model(image,forward_pass='features')
                    preds = model(image)
                    
                else:
                    image = image.to(device,non_blocking=True)
                    feats = model(image,forward_pass='features')
                    preds = model(feats,forward_pass=forwarding)


                features.append(feats)
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
        self.confidences = self.confidence_tensor

        self.classes = [ class_names[l.item()] for l in self.label_tensor  ]
        self.class_names = class_names

        self.dataset_size = self.label_tensor.shape[0]

        self.feature_tensor = torch.nn.functional.normalize(self.feature_tensor, dim = 1)

        y_train = self.label_tensor.detach().cpu().numpy()
        pred = self.prediction_tensor.detach().cpu().numpy()
        max_label = max(y_train)


        self.C = get_cost_matrix(pred, y_train, max_label+1)
        ri, ci = assign_classes_hungarian(self.C)

        self.cluster_to_class = torch.Tensor(ci)
        self.cluster_names = [ str(i)+'('+class_names[ci[i]]+')' for i in range(len(class_names)) ]
        self.class_of_prediction = self.cluster_to_class[self.prediction_tensor]
        self.clusters = [str(int(self.prediction_tensor[i]))+'('+class_names[int(self.class_of_prediction[i].item())]+')' for i in range(self.dataset_size) ]

        self.correct_samples = self.class_of_prediction == self.label_tensor
        self.bad_samples = self.correct_samples == False

        self.kNN_indices = None
        self.kNN_labels = None
        self.kNN_consistent = None
        self.kNN_confidences = None
        self.proximity = None # mean distance of the top nearest neighbors
        self.local_consistency = None
        self.criterion_consistent = None
        self.summary = None
        self.knn = 0

    def initialize_unlabeled(self,device,model,dataloader,forwarding='head',model_type='cluster_head'):
         
        model.eval()
        predictions = []
        features = []
        soft_labels = []
        confidences = []


        model = model.to(device)


        with torch.no_grad():      
            for batch in dataloader:
                if isinstance(batch,dict):
                    image = batch['image']
                else:
                    image = batch[0]

                if model_type == 'contrastive_clustering':
                    image = image.to(device,non_blocking=True)
                    feats, _,preds, _ = model(image,image)
    
                elif model_type == 'fixmatch_model':
                    image = image.to(device,non_blocking=True)
                    feats = model(image,forward_pass='features')
                    preds = model(image)
                    
                else:
                    image = image.to(device,non_blocking=True)
                    feats = model(image,forward_pass='features')
                    preds = model(feats,forward_pass=forwarding)


                features.append(feats)
                soft_labels.append(preds)
                max_confidence, prediction = torch.max(preds,dim=1) 
                predictions.append(prediction)
                confidences.append(max_confidence)


        self.feature_tensor = torch.cat(features)
        self.softlabel_tensor = torch.cat(soft_labels)
        self.prediction_tensor = torch.cat(predictions)
        self.confidence_tensor = torch.cat(confidences)
        self.confidences = self.confidence_tensor

        self.dataset_size = self.prediction_tensor.shape[0]
        self.feature_tensor = torch.nn.functional.normalize(self.feature_tensor, dim = 1)

    def compute_kNN_statistics(self,knn):
        self.knn = knn

        if self.dataset_size > 15000:

            idx_list = []
            for i in range(len(self.feature_tensor)):
                feature = torch.unsqueeze(self.feature_tensor[i],dim=0)
                similarities = torch.mm(feature,self.feature_tensor.t())
                scores, idx_ = similarities.topk(k=knn, dim=1)
                idx_list.append(idx_)
            idx_k = torch.cat(idx_list)

            labels_topk = torch.zeros_like(idx_k)
            confidence_topk = torch.zeros_like(idx_k,dtype=torch.float)
            for s in range(knn):
                labels_topk[:, s] = self.prediction_tensor[idx_k[:, s]]
                confidence_topk[:, s] = self.confidence_tensor[idx_k[:, s]]
            
            self.kNN_consistent = labels_topk[:, 0:1] == labels_topk # <boolean mask>
            self.local_consistency = self.kNN_consistent.sum(dim=1)/knn
            self.kNN_labels = labels_topk
            self.kNN_confidences = confidence_topk
            
        else:

            similarity_matrix = torch.einsum('nd,cd->nc', [self.feature_tensor.cpu(), self.feature_tensor.cpu()])
            scores_k, idx_k = similarity_matrix.topk(k=knn, dim=1)
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
             
    def compute_real_consistency(self, criterion):

        self.criterion_consistent = []
        for i in range(self.dataset_size):
            confids = self.kNN_confidences[i][self.kNN_consistent[i]] 
            real = confids > criterion
            self.criterion_consistent.append(sum(real)/self.knn)

        self.criterion_consistent = torch.Tensor(self.criterion_consistent)

    def return_statistic_summary(self,best_loss):
        statistic = pd.Series()

        statistic['Loss'] = best_loss
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

     
        statistic['cluster_entropy'] = cluster_metric['entropy']

        statistic['consistency_ratio'] = self.num_of_consistents(upper=0.9,lower=1.0,real_consistent=True)/self.dataset_size
        statistic['confidence_ratio'] = self.num_of_confidents(0.99)/self.dataset_size

        statistic['correct_mean_confidence'] = self.mean_from_selection(self.correct_samples.cpu().numpy(),self.confidence_tensor.cpu().numpy())
        statistic['bad_mean_confidence'] = self.mean_from_selection(self.bad_samples.cpu().numpy(),self.confidence_tensor.cpu().numpy())
        statistic['correct_mean_consistency'] = self.mean_from_selection(self.correct_samples.cpu().numpy(),self.criterion_consistent.cpu().numpy())
        statistic['bad_mean_consistency'] = self.mean_from_selection(self.bad_samples.cpu().numpy(),self.criterion_consistent.cpu().numpy())

        self.summary = statistic

        return statistic
    
    def get_accuracy(self):

        return to_value(sum(self.correct_samples)/len(self.correct_samples))
        

    def meanConfidence_of_consistent_kNN(self):
        
        means = []
        for i in range(self.dataset_size):
            confids = self.kNN_confidences[i][self.kNN_consistent[i]] 
            mean_confidence = sum(confids)/len(confids) 
            means.append(mean_confidence)

        return torch.Tensor(means)
        

    def meanConsistency_of_confidents(self,criterion):


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
                      
        
    def accuracy_from_selection(self,selection_mask,preselect_mask=None):

        #0# §(preselect_mask) --> correct_indicator
        if preselect_mask is None:
            correct_indicator = self.correct_samples
        else:
            correct_indicator = self.correct_samples[preselect_mask]
        ###

        correct_subset = correct_indicator[selection_mask]

        if len(correct_subset) < 1: return nan 

        return to_value(sum(correct_subset)/len(correct_subset))
        
        
    def ratio_from_selection(self,selection_mask,boolean_tensor,preselect_mask=None,relative=True):
        
        if preselect_mask is None: boolean_selected = boolean_tensor
        else: boolean_selected = boolean_tensor[preselect_mask]

        indicators = boolean_selected[selection_mask]
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

        category_values = self.select_mask(category_mapping,selection_mask) # ADAPT !

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

            if isinstance(c,torch.Tensor):
                category_names.append(str(c.item()))
            else: category_names.append(str(c))
            counting_mask = self.match_value(category_values,c) # outputs a Tensor of size category_values with True if value matches
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


    def mean_from_selection(self,selection_mask,scalar_tensor,preselect_mask=None):
        
        if selection_mask is None:
            selection_mask = torch.full([self.dataset_size],True)

        if preselect_mask is None:  scalar_select = scalar_tensor
        else: scalar_select = scalar_tensor[preselect_mask]

        subset_values = scalar_select[selection_mask]
        subset_size = len(subset_values)
        mean = sum(subset_values)/len(subset_values)

        if subset_size == 0: return 0

        return to_value(mean)
      

    def scalar_statistics_from_selection(self,selection_mask,scalar_tensor,scalar_name='',val_range=[0,1],interval=0.1,returntype='pandas',values_to_count=None):

        # init ------------------------------------

        if selection_mask is None:
            selection_mask = torch.full([self.dataset_size],True)
        
        if values_to_count: 
            measurements = values_to_count[selection_mask]
        else: measurements = None

        subset_values = scalar_tensor[selection_mask]
        subset_size = len(subset_values)

        # initialize index or column names
        names = []
        values = []
        iv = val_range[0]
        names.append(scalar_name+"|{:4.2f}-{:4.2f}".format(iv,iv+interval))
        iv += interval

        while iv <= val_range[1]:            
            names.append( "{:4.2f}-{:4.2f}".format(iv,iv+interval) )
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
            counted_value = self.count_set(cmask,measurements,'ratio') # count_mode
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
            subset_size = self.dataset_size

        else:
            selected_features = scalar_features[dataset_mask]
            subfeature_values = self.select_mask(second_feature,dataset_mask)
            subset_size = len(subfeature_values)

        val_range = parameters['range']
        interval = parameters['interval']
        #returntype = parameters['return_type'] # pandas or dictionary

        
        if subset_size == 0: 
            selected_features = torch.Tensor([0])
            subfeature_values = torch.Tensor([0])

        #return_info = {}
        names = []
        #values = [0]
        #numbers = [0]

        #high = max(scalar_tensor)
        iv = val_range[0]

        names.append(parameters['row_name']+"|{:4.2f}-{:4.2f}".format(iv,iv+interval))
        iv += interval
        
        while iv <= val_range[1]:            
            names.append("{:4.2f}-{:4.2f}".format(iv,iv+interval))
            iv += interval

        
        #parameters['secondary'] = True

        if 'count_measure' in parameters.keys():
            valuesToCount = parameters['count_measure']
            # values_to_count is in DATASET-SIZE, so it must be adapted
            vtc = self.select_mask(valuesToCount,dataset_mask) if dataset_mask is not None else valuesToCount
        else: vtc = None

        iv = val_range[0] - interval
        ni = 0

        while iv+interval <= val_range[1]:

            next_set = self.double_condition_mask(iv,iv+interval,selected_features)

            if parameters['secondary_type'] == 'categorical':
                next_row = self.categorical_from_selection(subfeature_values,next_set,values_to_count=vtc,mode=parameters['mode'],return_type=parameters['return_type'])
            elif parameters['secondary_type'] == 'scalar_interval':
                next_row = self.scalar_statistics_from_selection(next_set,subfeature_values,scalar_name=parameters['col_name'],val_range=parameters['range_2'],interval=parameters['interval_2'],returntype='pandas',values_to_count=vtc)
            elif parameters['secondary_type'] == 'scalar_cumulative':
                next_row = self.scalar_cumulative_from_selection(scalar_tensor=subfeature_values,selection_mask=next_set,func='fraction',scalar_name=parameters['col_name'],val_range=parameters['range_2'],interval=parameters['interval_2'],returntype='pandas',values_to_count=vtc)
            else:
                raise ValueError("invalid property type")
            
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
            subset_size = self.dataset_size

        else:
            selected_features = self.select_mask(category_features,dataset_mask)
            subfeature_values = self.select_mask(second_feature,dataset_mask)
            subset_size = len(subfeature_values)

        
        #returntype = parameters['return_type'] # pandas or dictionary
       
        if subset_size == 0: 
            selected_features = torch.Tensor([0])
            subfeature_values = torch.Tensor([0])

        if isinstance(category_features,torch.Tensor):
            bins = torch.unique(category_features)
        else: bins = set(category_features)
        

        second_type = parameters['secondary_type']

        if second_type == 'categorical':
        
            
            mode = parameters['mode'] 
            rtype = parameters['return_type']
            
            if 'count_measure' in parameters.keys():
                valuesToCount = parameters['count_measure']
                # values_to_count is in DATASET-SIZE, so it must be adapted
                vtc = self.select_mask(valuesToCount,dataset_mask) if dataset_mask is not None else valuesToCount
            else: vtc = None
                

            for v in bins:
                subcategory_mask = self.match_value(selected_features,v)
                rowSeries = self.categorical_from_selection(subfeature_values,subcategory_mask,values_to_count=vtc,mode=mode,return_type=rtype)
                added = pd.DataFrame(dict(rowSeries),columns=rowSeries.index,index=[str(v)])
                result_frame = pd.concat([result_frame,added])
                #result_frame = pd.concat([result_frame,rowSeries])

        elif second_type == 'scalar_interval':
            #valuesToCount = parameters['']
            if 'count_measure' in parameters.keys(): 
                values_to_count = parameters['count_measure']
                if dataset_mask is not None: parameters['count_measure'] = self.select_mask(values_to_count,dataset_mask)
            else: parameters['count_measure'] = None
            
            for v in bins:
                subcategory_mask = self.match_value(selected_features,v)
                rowSeries = self.scalar_statistics_from_selection(subcategory_mask,subfeature_values,scalar_name=parameters['col_name'],val_range=parameters['range_2'],interval=parameters['interval_2'],returntype='pandas',values_to_count=parameters['count_measure'])
                added = pd.DataFrame(dict(rowSeries),columns=rowSeries.index,index=[str(v)])
                result_frame = pd.concat([result_frame,added])

        elif second_type == 'scalar_cumulative':
            if 'count_measure' in parameters.keys(): 
                values_to_count = parameters['count_measure']
                if dataset_mask is not None: parameters['count_measure'] = self.select_mask(values_to_count,dataset_mask)
            else: parameters['count_measure'] = None

            for v in bins:
                subcategory_mask = self.match_value(selected_features,v)
                rowSeries = self.scalar_cumulative_from_selection(scalar_tensor=subfeature_values,selection_mask=subcategory_mask,func='fraction',scalar_name=parameters['col_name'],val_range=parameters['range_2'],interval=parameters['interval_2'],returntype='pandas',values_to_count=parameters['count_measure'])
                added = pd.DataFrame(dict(rowSeries),columns=rowSeries.index,index=[str(v)])
                result_frame = pd.concat([result_frame,added])
        else: 
            raise ValueError('invalid type for second dimension')
            

        return result_frame


    def metric_of_categorical_from_selection(self,category_values,dataset_mask=None,func="accuracy",value_base=None,return_type='pandas'):

        if dataset_mask is None:
            selected_features = category_values
            subset_size = self.dataset_size
            #preselect_mask = None
        else:
            selected_features = self.select_mask(category_values,dataset_mask)
            subset_size = len(selected_features)
            #preselect_mask = dataset_mask

        if isinstance(category_values,torch.Tensor):
            bins = torch.unique(category_values)
        else: bins = set(category_values)

        category_names = []
        amounts = []

        for v in bins:
            if isinstance(v,torch.Tensor):
                category_names.append(str(v.item()))
            else: category_names.append(str(v))

            subcategory_mask = self.match_value(selected_features,v)
            if func == 'accuracy':
                amounts.append( self.accuracy_from_selection(subcategory_mask,dataset_mask) )
            elif func == 'mean':
                amounts.append( self.mean_from_selection(subcategory_mask,value_base,dataset_mask) )
            elif func == 'ratio':
                amounts.append( self.ratio_from_selection(subcategory_mask,value_base,dataset_mask) )
            else: raise ValueError('not yet implemented')

        if return_type == 'list': 
            return category_names, amounts

        elif return_type == 'pandas':
            result = pd.Series(amounts,index=category_names)
            return result
            

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
            if ('ratio' in count_mode) or ('mean' in count_mode):
                return to_value(sum(set_mask)/len(set_mask))
            else: return to_value(sum(set_mask))

        else: 
            if ('ratio' in count_mode) or ('mean' in count_mode):
                feature_values = measurements[set_mask]
                return to_value(sum(feature_values)/len(feature_values))
            else: return to_value(sum(feature_values))


    def select_mask(self,category_values,selection_mask):
 
        if isinstance(category_values,torch.Tensor):
            return category_values[selection_mask]
        else:
            result = []
            for i in range(len(selection_mask)):
                if selection_mask[i]: result.append(category_values[i])        
            return result


    def match_value(self,values,query):

        if isinstance(query,str):
            result = torch.full([len(values)],False)
            for i in range(len(values)):
                result[i] = values[i] == query
            return result
        else:
            return values == query


#@doc: define a subset of the dataset from all samples with a chosen criterion --> selction_mask
# We evaluate the real_value property from [scalar_tensor] to evaluate the number or ratio of samples with a certain property value
# within the selected samples of the mask. 
    def scalar_cumulative_from_selection(self,scalar_tensor,selection_mask=None,func='accuracy',scalar_name='',val_range=[0,1.0],interval=0.1,returntype='pandas',values_to_count=None):

        # init -----------------------------------
        measurements = None

        if selection_mask is None:
            #selection_mask = torch.full([self.dataset_size],True)
            subset_values = scalar_tensor
            if values_to_count: measurements = values_to_count
        else:
            subset_values = scalar_tensor[selection_mask]
            if values_to_count: measurements = values_to_count[selection_mask]

        subset_size = len(subset_values)


        # initialize index or column names
        names = []
        values = []
        iv = val_range[0]
        names.append(scalar_name+"|>{:4.2f}".format(iv))
        iv += interval

        while iv <= val_range[1]:            
            names.append( ">{:4.2f}".format(iv) )
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
            cmask =  subset_values >= iv+interval
            if func == 'accuracy':
                values.append( self.accuracy_from_selection(cmask, selection_mask) )
            elif func == 'ratio':
                values.append( self.ratio_from_selection(cmask, values_to_count, selection_mask) )
            else:
                # equivalent to mean_from_selection
                counted_value = self.count_set(cmask,measurements,'ratio') # count_mode
                values.append(counted_value)
            numbers.append(to_value(sum(cmask)))
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


    def two_cumulative_from_selection(self,scalar_tensor,second_feature,parameters,dataset_mask=None):


        result_frame = pd.DataFrame()

        if dataset_mask is None:
            selected_features = scalar_tensor
            subfeature_values = second_feature
            subset_size = self.dataset_size

        else:
            selected_features = scalar_tensor[dataset_mask]
            subfeature_values = self.select_mask(second_feature,dataset_mask)
            subset_size = len(subfeature_values)

        val_range = parameters['range']
        interval = parameters['interval']
        #returntype = parameters['return_type'] # pandas or dictionary

        
        if subset_size == 0: 
            selected_features = torch.Tensor([0])
            subfeature_values = torch.Tensor([0])

        scalar_name = parameters['row_name']

        names = []
        #values = []
        iv = val_range[0]
        names.append(scalar_name+"|>{:4.2f}".format(iv))
        iv += interval

        while iv <= val_range[1]:            
            names.append( ">{:4.2f}".format(iv) )
            iv += interval

        
        #parameters['secondary'] = True

        if 'count_measure' in parameters.keys():
            valuesToCount = parameters['count_measure']
            # values_to_count is in DATASET-SIZE, so it must be adapted
            vtc = self.select_mask(valuesToCount,dataset_mask) if dataset_mask is not None else valuesToCount
        else: vtc = None

        iv = val_range[0] - interval
        ni = 0

        while iv+interval <= val_range[1]:

            next_set = selected_features >= iv+interval

            if parameters['secondary_type'] == 'categorical':
                next_row = self.categorical_from_selection(subfeature_values,next_set,values_to_count=vtc,mode=parameters['mode'],return_type=parameters['return_type'])
            elif parameters['secondary_type'] == 'scalar_interval':
                next_row = self.scalar_statistics_from_selection(next_set,subfeature_values,scalar_name=parameters['col_name'],val_range=parameters['range_2'],interval=parameters['interval_2'],returntype='pandas',values_to_count=vtc)
            elif parameters['secondary_type'] == 'scalar_cumulative':
                next_row = self.scalar_cumulative_from_selection(scalar_tensor=subfeature_values,selection_mask=next_set,func='fraction',scalar_name=parameters['col_name'],val_range=parameters['range_2'],interval=parameters['interval_2'],returntype='pandas',values_to_count=vtc)
            else:
                raise ValueError("invalid property type")
            
            added = pd.DataFrame(dict(next_row),columns=next_row.index,index=[names[ni]])
            result_frame = pd.concat([result_frame,added])
            iv += interval
            ni += 1
    

        return result_frame



#@author: Michael Blachetta
class logger():
    def __init__(self,value={},unit_name='MERGE_unsupervisedClustering',unit_type='<APPLICATION>'):

        self.type = unit_type
        self.name = unit_name
        self.properties = value
        self.elements = []

        self.head_text = []

    def add_element(self, element):
        self.elements.append(element)

    def set_values(self,value):
        self.properties = value

    def add_value(self,key,value):
        self.properties[key] = value

    def __str__(self):
        out = self.type+' '+self.name+':\n'
        for k in self.properties.keys():
            out += k+' = '+str(self.properties[k])+'\n'

        out += '\n'
        for element in self.elements:
            out += element.outstr('  ')

        return out

    def outstr(self,indent=''):
        
        out = indent+self.type+' '+self.name+':\n'
        for k in self.properties.keys():
            out += indent+k+' = '+str(self.properties[k])+'\n'

        out += '\n'
        for element in self.elements:
            out += element.outstr(indent+'  ')

        return out

    def unit_str(self):
        
        out = ''
        d = ''
        for t in self.head_text:
            out += t
            d += '  '

        out = d+self.type+' '+self.name+':\n'
        for k in self.properties.keys():
            out += d+k+' = '+str(self.properties[k])+'\n'

        return out

    def full_str(self):
        
        out = ''
        d = ''
        for t in self.head_text:
            out += t
            d += '  '

        out = d+self.type+' '+self.name+':\n'
        for k in self.properties.keys():
            out += d+k+' = '+str(self.properties[k])+'\n'

        out += '\n'
        for element in self.elements:
            out += element.outstr(d+'  ')

        return out

    def head_str(self,len_hlist=0):

        d = ' '*len_hlist
        out = d+self.type+' '+self.name+':\n'
        for k in self.properties.keys():
            out += d+k+' = '+str(self.properties[k])+'\n'

        return out

    def add_head_text(self,text):
        self.head_text.append(text)

    def __len__(self):
        return len(self.head_text)

    def to_file(self,path,method='full_str'):
        report = ''

        if method == 'full_str':
            report = self.full_str()
        elif method == 'unit_str':
            report = self.unit_str()
        elif method == 'outstr':
            report = self.outstr()
        else: raise ValueError('invalid log method')

        with open(path,'w') as f:
            f.write(report)

