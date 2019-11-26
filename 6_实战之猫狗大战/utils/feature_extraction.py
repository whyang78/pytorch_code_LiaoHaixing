import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import argparse
import h5py
from net import feature_net

parse=argparse.ArgumentParser(description='feature extraction with different models')
parse.add_argument('--model',required=True,help='vgg16,inceptionv3,resnet18')
parse.add_argument('--batch_size',type=int,default=5)
parse.add_argument('--phase',required=True,help='train,val')
opt=parse.parse_args()

feature_path='../feature'
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

use_gpu=torch.cuda.is_available()
train_path='../../dataset/cat_vs_dog/data/train'
val_path='../../dataset/cat_vs_dog/data/val'

trainData=ImageFolder(root=train_path,transform=transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]))
trainDataLoader=DataLoader(trainData,batch_size=opt.batch_size,shuffle=False,num_workers=4)

valData=ImageFolder(root=val_path,transform=transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]))
valDataLoader=DataLoader(valData,batch_size=opt.batch_size,shuffle=False,num_workers=4)

data={'train':trainData,'val':valData}
dataloader={'train':trainDataLoader,'val':valDataLoader}

def create_feature(model,phase,feature_path):
    net=feature_net(model)
    if use_gpu:
        net.cuda()

    feature_map=torch.FloatTensor()
    label_map=torch.LongTensor()
    for batchdata,batchlabel in dataloader[phase]:
        if use_gpu:
            batchdata=batchdata.cuda()
        output=net(batchdata)
        feature_map=torch.cat((feature_map,output.cpu().data),dim=0)
        label_map=torch.cat((label_map,batchlabel),dim=0)
    print(feature_map.shape)
    feature_map=feature_map.numpy()
    label_map=label_map.numpy()
    h5_path=os.path.join(feature_path,phase+'_feature_{}'.format(model))
    with h5py.File(h5_path,'w') as h:
        h.create_dataset('data',data=feature_map)
        h.create_dataset('label',data=label_map)


if __name__ == '__main__':
    create_feature(opt.model,opt.phase,feature_path)