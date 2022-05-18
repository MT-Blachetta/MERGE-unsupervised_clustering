import torch
import numpy as np
from sklearn.utils import shuffle
from sklearn import cluster
import sklearn
from sklearn.decomposition import IncrementalPCA
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment





def batches(l, n):
    for i in range(0, len(l), n): # step_size = n, [i] is a multiple of n
        yield l[i:i + n] # teilt das array [l] in batch sub_sequences der Länge n auf




def get_cost_matrix(y_pred, y, nc=1000): # C[ground-truth_classes,cluster_labels] counts all instances with a given ground-truth and cluster_label
    C = np.zeros((nc, y.max() + 1))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
    return C 



def assign_classes_hungarian(C): # rows are (num. of) clusters and columns (num. of) ground-truth classes
    row_ind, col_ind = linear_sum_assignment(C, maximize=True) # assume 1200 rows(clusters) and 1000 cols(classes)
    ri, ci = np.arange(C.shape[0]), np.zeros(C.shape[0]) # ri contains all CLASS indexes as integer from 0 --> num_classes
    ci[row_ind] = col_ind # assignment of the col_ind[column nr. = CLASS_ID] to the [row nr. = cluster_ID/index]

    # for overclustering, rest is assigned to best matching class
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False # True = alle cluster die nicht durch [linear_sum_assignment] einer Klasse zugeordnet wurden
    ci[mask] = C[mask, :].argmax(1) # Für weitere Cluster über die Anzahl Klassen hinaus, ordne die Klasse mit der größten Häufigkeit zu 
    return ri.astype(int), ci.astype(int) # at each position one assignment: ri[x] = index of cluster <--> ci[x] = classID assigned to cluster


def assign_classes_majority(C):
    col_ind = C.argmax(1) # assign class with the highest occurence to the cluster (row)
    row_ind = np.arange(C.shape[0]) # clusterID at position in both arrays (col_ind and row_ind)

    # best matching class for every cluster
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False

    return row_ind.astype(int), col_ind.astype(int)



#cluster_idx,class_idx = assign_classes_hungarian(C_train)
#rid,cid = assign_classes_majority(C_train)

def accuracy_from_assignment(C, row_ind, col_ind, set_size=None):
    if set_size is None:
        set_size = C.sum()
    cnt = C[row_ind, col_ind].sum() # sum of all correctly (class)-assigned instances that contributes to the Cluster's ClassID decision
    # (that caused the decision)
    return cnt / set_size # If all clusters would have only instaces of one unique class, this value becomes = 1

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
  
    return {'best_clusters': good_clusters, 'classes': best_class, 'accuracies': best_acc}


def train_pca(X_train,n_comp):
    bs = max(4096, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs,n_components=n_comp)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs + 1)):
        transformer = transformer.partial_fit(batch)
        # break
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer

def transform_pca(X, transformer):
    n = max(4096, X.shape[1] * 2)
    n_comp = transformer.components_.shape[0]
    X_ = np.zeros((X.shape[0],n_comp))
    for i in trange(0, len(X), n):
        X_[i:i + n] = transformer.transform(X[i:i + n])
        # break
    return X_


@torch.no_grad()
def evaluate_singleHead(device,model,dataloader,formatation=False):

    model.eval()
    predictions = []
    labels = []

    model.to(device)
    #type_test = next(iter(dataloader))
    #isinstance

    for batch in dataloader:
        if isinstance(batch,dict):
            image = batch['image']
            labels = batch['target']
        else:
            image = batch[0]
            labels = batch[1]

        image = image.to(device,non_blocking=True)
        predic = model(image,forward_pass='eval')
        predictions.append(torch.argmax(predic, dim=1))
        labels.append(label)

    prediction_tensor = torch.cat(predictions)
    label_tensor = torch.cat(labels)
    
    y_train = label_tensor.detach().cpu().numpy()
    pred = prediction_tensor.detach().cpu().numpy()
    max_label = max(y_train)
    assert(max_label==9)

    C_train = get_cost_matrix(pred, y_train, max_label+1)

    message = 'val'
    y_pred = pred
    y_true = y_train
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

    print("\n{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}\tACC {:.5e}".format(message, ari, v_measure, ami, fm, acc_tr_lin))

    return result_dict


def evaluate_prediction(y_true,y_pred,formatation=False):

    max_label = max(y_true)
    assert(max_label==9)
    print(y_true)
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



def evaluate_headlist(device,model,dataloader,formatation=False):

    predictions = [ [] for _ in range(model.nheads) ]
    label_list = []
    #labels = [ [] for _ in range(model.nheads)]
    model.to(device)

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
        print(rdict['ACC'])
        dicts.append(rdict)

    best_head = np.argmax(np.array(accuracies))

    result_dict = dicts[best_head]
    result_dict['head_id'] = best_head

    acc_tr_lin = result_dict['ACC'] 
    ari = result_dict['ARI'] 
    v_measure = result_dict['V_measure']
    fm = result_dict['fowlkes_mallows']
    ami = result_dict['AMI']

    message = 'validation'

    print('best head is ',best_head)
    print("\n{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}\tACC {:.5e}".format(message, ari, v_measure, ami, fm, acc_tr_lin))

    return result_dict
        

    


