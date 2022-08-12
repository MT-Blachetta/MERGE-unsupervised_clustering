from resnet import ResNet, get_resnet
from models import Network
from datasets import STL10_eval, STL10
from functionality import collate_custom
import torch
from evaluate import Analysator
import torchvision.transforms as transforms



checkpoint = torch.load('/home/blachm86/unsupervisedClustering/SELFLABEL/FINAL_MODEL_model.pth',map_location='cpu')
#model_dict = checkpoint['net']
model_dict = checkpoint

backbone = get_resnet("ResNet50")
model = Network(backbone,128,10)
model.load_state_dict(model_dict,strict=True)




val_transformations = transforms.Compose([
                                transforms.CenterCrop(96),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
#eval_dataset = STL10_eval(path='/space/blachetta/data',aug=val_transformations)
eval_dataset = STL10(split='train+unlabeled',transform=val_transformations)

val_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=8,
                batch_size=256, pin_memory=True, collate_fn=collate_custom,
                drop_last=False, shuffle=False)


print('compute features in Analysator')

eval_object = Analysator('cuda:3',model,val_dataloader,forwarding='head',model_type='fixmatch_model',labeled=False)

eval_object.compute_kNN_statistics(200) # 100 for labeled dataset
eval_object.compute_real_consistency(0.5)
#results = eval_object.return_statistic_summary(0)
#print(results)
print('ready !')

torch.save(eval_object,'/home/blachm86/train+unlabeled_analysator.torch')
