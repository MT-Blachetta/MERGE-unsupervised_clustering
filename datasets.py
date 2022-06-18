from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from torch.utils.data import Dataset
from utils.mypath import MyPath
import os
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob
import os
import pickle
import sys
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from functionality import collate_custom
from evaluate import get_cost_matrix, assign_classes_hungarian, accuracy_from_assignment

class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None

        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.classes = dataset.classes

        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['image'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output

class ReliableSamplesSet(Dataset): # §: ReliableSamplesSet_Initialisation

    def __init__(self,dataset,eval_transform):
        self.dataset = dataset
        self.index_mapping = []
        self.predictions = None
        self.dsize = 1
        self.transform = self.dataset.transform
        #print('self.transform: ',self.transform )
        self.eval_transform = eval_transform
        self.num_clusters = 0

        self.confidence = None
        self.alternative_consistency = None
        self.consistency = None


    def evaluate_samples(self,p,model,forwarding='head',knn=100):

        device = p['device']
        self.dataset.transform = self.eval_transform

        #print('dataset[0].shape ',type(self.dataset[0]))
        #print('dataset[0].shape ',self.dataset[0].shape)


        val_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=p['num_workers'],
                                                    batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                                    drop_last=False, shuffle=False)

        #testbatch = next(iter(val_dataloader))
        #print('testbatch-type: ',type(testbatch))
        #print('testbatch-len: ',len(testbatch))
        #print('testbatch-shape: ',testbatch.shape)

        model.eval()
        predictions = []
        labels = []
        features = []
        soft_labels = []
        confidences = []

        model = model.to(device) # OK(%-cexp_00)
        #print('MODEL first critical expresssion: ',type(model))


        with torch.no_grad():      
            for batch in val_dataloader:
                if isinstance(batch,dict):
                    image = batch['image']
                    label = batch['target']
                else:
                    image = batch[0]
                    label = batch[1]

                image = image.to(device,non_blocking=True) # OK(%-cexp_00)
                fea = model(image,forward_pass='features')
                features.append(fea)
                preds = model(fea,forward_pass=forwarding)
                soft_labels.append(preds)
                max_confidence, prediction = torch.max(preds,dim=1) 
                predictions.append(prediction)
                confidences.append(max_confidence)
                labels.append(label)

        feature_tensor = torch.cat(features)
        #self.softlabel_tensor = torch.cat(soft_labels)
        self.predictions = torch.cat(predictions)
        #print('max_prediction A: ',self.predictions.max())
        self.predictions = self.predictions.type(torch.LongTensor)
        #print('max_prediction B: ',self.predictions.max())
        #print('len(self.predictions) B: ',len(self.predictions))
        self.num_clusters = self.predictions.max()+1 # !issue: by test config assert(self.num_clusters == 10) get 9
        #print('num_clusters: ',self.num_clusters)
        self.label_tensor = torch.cat(labels)
        self.confidence = torch.cat(confidences)
        dataset_size = len(self.dataset)

        # §_Compute_Accuracy-------------------------------
        y_train = self.label_tensor.detach().cpu().numpy()
        pred = self.predictions.detach().cpu().numpy()
        max_label = max(y_train)
        #assert(max_label==9)
        C = get_cost_matrix(pred, y_train, max_label+1)
        ri, ci = assign_classes_hungarian(C)
        accuracy = accuracy_from_assignment(C,ri,ci)
        print('Accuracy: ',accuracy)
        # §------------------------------------------------

        feature_tensor = torch.nn.functional.normalize(feature_tensor, dim = 1)
        similarity_matrix = torch.einsum('nd,cd->nc', [feature_tensor, feature_tensor]) # removed .cpu()

        #self.knn = knn
        scores, idx_k = similarity_matrix.topk(k=knn, dim=1)
        #self.proximity = torch.mean(scores_k,dim=1)
        #self.kNN_indices = idx_k
        labels_topk = torch.zeros_like(idx_k)
        confidence_topk = torch.zeros_like(idx_k,dtype=torch.float)
        for s in range(knn):
            labels_topk[:, s] = self.predictions[idx_k[:, s]]
            confidence_topk[:, s] = self.confidence[idx_k[:, s]]
        
        kNN_consistent = labels_topk[:, 0:1] == labels_topk # <boolean mask>
        #kNN_labels = labels_topk
        kNN_confidences = confidence_topk
        criterion_consistent = []
        for i in range(dataset_size):
            confids = kNN_confidences[i][kNN_consistent[i]] # +logical_index > +true for index of consistent label; +size=knn > +indexes topk instances
            real = confids > 0.5
            criterion_consistent.append(sum(real)/knn)

        self.alternative_consistency = kNN_consistent.sum(dim=1)/knn

        self.consistency = torch.Tensor(criterion_consistent)
        self.select_top_samples()
        self.dataset.transform = self.transform
        self.top_samples_accuracy()
# §

    def __len__(self):
        return self.dsize


    def __getitem__(self, index):
        
        lx = self.index_mapping[index]
        out = self.dataset.__getitem__(lx)
        imagelist = out['image']
        target = self.predictions[lx]

        if isinstance(imagelist,list):
            return {'image': imagelist[0], 'target': target}
        else:
            return {'image':imagelist, 'target': target}


    def select_top_samples(self):

        print('len(dataset): ',len(self.dataset))
        min_size = 3 #(len(self.dataset)/self.num_clusters)*0.1
        print('min_size = ',min_size)
        start_ratio_cf = 0.99
        #start_ratio_cs = 0.99

        confident_samples, num_confident = self.get_confident_samples(ratio=start_ratio_cf)
        confirmed_samples, num_confirmed = self.get_consistent_samples(confident_samples)  # ,start_ratio_cs)

        while((num_confident < min_size) and (num_confirmed < 1)):
            
            start_ratio_cf -= 0.005
            print('reduce confidence to ', start_ratio_cf)
            confident_samples, num_confident = self.get_confident_samples(ratio=start_ratio_cf)
            confirmed_samples, num_confirmed = self.get_consistent_samples(confident_samples)
        
        self.index_mapping = []

        for label_samples in confirmed_samples:
            self.index_mapping.extend([ s.item() for s in list(label_samples[:num_confirmed]) ])

        self.dsize = len(self.index_mapping)
        print('reliableSamples_size = ',self.dsize)

        # num confirmed kommt in den index [:]


    def get_confident_samples(self,ratio):

        confirmed_samples = []
        min_confirmed = 100000000

        for label in range(self.num_clusters):
            label_indices = torch.where(self.predictions == label)[0]
            label_confidence = self.confidence[self.predictions == label]
            label_confirmed = label_indices[label_confidence > ratio]
            label_confirmed = label_confirmed.type(torch.LongTensor)
            confirmed_samples.append(label_confirmed)
            num_confirmed = len(label_confirmed)
            if num_confirmed < min_confirmed: min_confirmed = num_confirmed

        return confirmed_samples, min_confirmed            
        

    def get_consistent_samples(self,confident_samples):

        ratio = 0.99

        min_size = 3 #(len(self.dataset)/self.num_clusters)*0.1
        min_consistency = 100000000
        confirmed_samples = []

        for indices in confident_samples:
            consistencies = self.consistency[indices]
            confirmed = indices[consistencies > ratio]
            confirmed_samples.append(confirmed)
            num_c = len(confirmed)
            if num_c < min_consistency: min_consistency = num_c


        while(min_consistency < min_size):

            ratio -= 0.01

            if ratio < 0.899: 
                confirmed_samples, min_consistency = self.get_alternative_consistence(confident_samples)
                if min_consistency < 1: return confirmed_samples, min_consistency
            else:
                min_consistency = 100000000
                confirmed_samples = []

                for indices in confident_samples:
                    consistencies = self.consistency[indices]
                    confirmed = indices[consistencies > ratio]
                    confirmed_samples.append(confirmed)
                    num_c = len(confirmed)
                    if num_c < min_consistency: min_consistency = num_c

        print('consistency_ratio: ',ratio)
        print('min_confirmed = ',min_consistency)
        return confirmed_samples, min_consistency

    def get_alternative_consistence(self,confident_samples):
        
        print('looking for alternative consistency criterion')
        ratio = 0.99

        min_size = 3#(len(self.dataset)/self.num_clusters)*0.1
        min_consistency = 100000000
        confirmed_samples = []

        for indices in confident_samples:
            consistencies = self.alternative_consistency[indices]
            confirmed = indices[consistencies > ratio]
            confirmed_samples.append(confirmed)
            num_c = len(confirmed)
            if num_c < min_consistency: min_consistency = num_c


        while(min_consistency < min_size):

            ratio -= 0.01

            if ratio < 0.899:
                print('minimum consistency tolerance reached')
                return confirmed_samples, -1
            else:
                min_consistency = 100000000
                confirmed_samples = []

                for indices in confident_samples:
                    consistencies = self.alternative_consistency[indices]
                    confirmed = indices[consistencies > ratio]
                    confirmed_samples.append(confirmed)
                    num_c = len(confirmed)
                    if num_c < min_consistency: min_consistency = num_c

        print('alternative_consistency_ratio: ',ratio)
        return confirmed_samples, min_consistency

        #index_of_confidents = torch.where(self.confidence > 0.99)[0]

    def top_samples_accuracy(self):
        
        labels = []
        predlist = []
        for i in self.index_mapping:
            out = self.dataset.__getitem__(i)
            labels.append(to_value(out['target']))
            predlist.append(to_value(self.predictions[i]))

        y_train = np.array(labels)
        pred = np.array(predlist)
        max_label = max(y_train)
        C = get_cost_matrix(pred, y_train, max_label+1)
        ri, ci = assign_classes_hungarian(C)
        accuracy = accuracy_from_assignment(C,ri,ci)
        print('top_samples accuracy: ',accuracy)

        



def to_value(v):
    if isinstance(v,torch.Tensor):
        v = v.item()        
    return v


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root=MyPath.db_root_dir('cifar-10'), train=True, transform=None, 
                    download=False):

        super(CIFAR10, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]        

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR20(CIFAR10):
    """CIFAR20 Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
                    ['train', '16019d7e3df5f24257cddd939b257f8d'],
                 ]

    test_list = [
                    ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
                ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    def __init__(self, root=MyPath.db_root_dir('cifar-20'), train=True, transform=None, 
                    download=False):
        super(CIFAR20, self).__init__(root, train=train,transform=train_transformation,
                                        download=download)
        # Remap classes from cifar-100 to cifar-20
        new_ = self.targets
        for idx, target in enumerate(self.targets):
            new_[idx] = _cifar100_to_cifar20(target)
        self.targets = new_
        self.classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 'household electrical devices', 'househould furniture', 'insects', 'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']


def _cifar100_to_cifar20(target):
  _dict = \
    {0: 4,
     1: 1,
     2: 14,
     3: 8,
     4: 0,
     5: 6,
     6: 7,
     7: 7,
     8: 18,
     9: 3,
     10: 3,
     11: 14,
     12: 9,
     13: 18,
     14: 7,
     15: 11,
     16: 3,
     17: 9,
     18: 7,
     19: 11,
     20: 6,
     21: 11,
     22: 5,
     23: 10,
     24: 7,
     25: 6,
     26: 13,
     27: 15,
     28: 3,
     29: 15,
     30: 0,
     31: 11,
     32: 1,
     33: 10,
     34: 12,
     35: 14,
     36: 16,
     37: 9,
     38: 11,
     39: 5,
     40: 5,
     41: 19,
     42: 8,
     43: 8,
     44: 15,
     45: 13,
     46: 14,
     47: 17,
     48: 18,
     49: 10,
     50: 16,
     51: 4,
     52: 17,
     53: 4,
     54: 2,
     55: 0,
     56: 17,
     57: 4,
     58: 18,
     59: 17,
     60: 10,
     61: 3,
     62: 2,
     63: 12,
     64: 12,
     65: 16,
     66: 12,
     67: 1,
     68: 9,
     69: 19,
     70: 2,
     71: 10,
     72: 0,
     73: 1,
     74: 16,
     75: 12,
     76: 9,
     77: 13,
     78: 15,
     79: 13,
     80: 16,
     81: 19,
     82: 2,
     83: 4,
     84: 6,
     85: 19,
     86: 5,
     87: 5,
     88: 8,
     89: 19,
     90: 18,
     91: 1,
     92: 2,
     93: 15,
     94: 6,
     95: 0,
     96: 17,
     97: 8,
     98: 14,
     99: 13}

  return _dict[target]



class ImageNet(datasets.ImageFolder):
    def __init__(self, root=MyPath.db_root_dir('imagenet'), split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, 'ILSVRC2012_img_%s' %(split)),
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img


class ImageNetSubset(data.Dataset):
    def __init__(self, subset_file, root=MyPath.db_root_dir('imagenet'), split='train', 
                    transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, 'ILSVRC2012_img_%s' %(split))
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i)) 
        self.imgs = imgs 
        self.classes = class_names
    
	# Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out



class STL10(Dataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        folds (int, optional): One of {0-9} or None.
            For training, loads one of the 10 pre-defined folds of 1k samples for the
             standard evaluation procedure. If no value is passed, loads the 5k samples.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(self, root=MyPath.db_root_dir('stl-10'),
                 split='train', folds=None, transform=None,
                 download=False):
        super(STL10, self).__init__()
        self.root = root
        self.transform = transform
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)
        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. '
                'You can use download=True to download it')

        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)

        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        if self.split == 'train': # Added this to be able to filter out fp from neighbors
            self.targets = self.labels


    def _verify_folds(self, folds):
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = ("Value for argument folds should be in the range [0, 10), "
                   "but got {}.")
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
            class_name = self.classes[target]
        else:
            img, target = self.data[index], 255 # 255 is an ignore index
            class_name = 'unlabeled'

        # make consistent with all other datasets
        # return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img_size = img.size

        if self.transform is not None:
            img = self.transform(img)
        
        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out

    def get_image(self, index):
        img = self.data[index]
        img = np.transpose(img, (1, 2, 0))
        return img

    def __len__(self):
        return self.data.shape[0]

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds):
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.uint8, sep=' ')
            self.data, self.labels = self.data[list_idx, :, :, :], self.labels[list_idx]

class STL10_trainNtest(torchvision.datasets.VisionDataset):
    def __init__(self,path,aug):
        self.aug = aug
        self.train_dataset = torchvision.datasets.STL10(path, split='train', download=False, transform=None)
        self.train_len = len(self.train_dataset)
        self.test_dataset = torchvision.datasets.STL10(path, split='test', download=False, transform=None)
        self.test_len = len(self.test_dataset)

    def __len__(self):
        return self.train_len + self.test_len

    def __getitem__(self,index):
        if index >= self.train_len:
            index -= self.train_len
            img, target = self.test_dataset[index]
        else:
            img, target = self.train_dataset[index]

        imgs = self.aug(img)

        return imgs, target


class STL10_eval(torchvision.datasets.VisionDataset):
    def __init__(self,path,aug):
        self.transform = aug
        self.train_dataset = torchvision.datasets.STL10(path, split='train', download=False, transform=None)
        self.train_len = len(self.train_dataset)
        self.test_dataset = torchvision.datasets.STL10(path, split='test', download=False, transform=None)
        self.test_len = len(self.test_dataset)
        self.classes = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

    def __len__(self):
        return self.train_len + self.test_len

    def __getitem__(self,index):
        if index >= self.train_len:
            index -= self.train_len
            img, target = self.test_dataset[index]
        else:
            img, target = self.train_dataset[index]

        if self.transform is not None:
            imgs = self.transform(img)
        else: imgs = img

        out = { 'image': imgs, 'target': target, 'meta': {'im_size': [3,96,96], 'index': index, 'class_name': self.classes[target]} }

        return out

        #return imgs, target