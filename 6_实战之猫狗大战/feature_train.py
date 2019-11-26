import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import os
# import sys
# sys.path.append('.')
from utils import h5_dataset,classifier

bs=32
epoch=10
lr=0.001
numclasses=2
use_gpu=torch.cuda.is_available()

models=['vgg16','inceptionv3','resnet18']
feature_path='./feature'
train_list=[os.path.join(feature_path,'train_feature_{}'.format(m))for m in models]
val_list=[os.path.join(feature_path,'val_feature_{}'.format(m))for m in models]

#加载数据集
trainData=h5_dataset(train_list)
valData=h5_dataset(val_list)
trainDataLoader=DataLoader(trainData,batch_size=bs,shuffle=True,num_workers=4)
valDataLoader=DataLoader(valData,batch_size=bs,shuffle=False,num_workers=4)

#构建网络
feature_dim=trainData.feature.size(1)
net=classifier(feature_dim,numclasses)
optimizer=optim.Adam(net.parameters(),lr=lr)
criterion=nn.CrossEntropyLoss()
if use_gpu:
    net.cuda()
    criterion=criterion.cuda()
    
if __name__ == '__main__':
    #训练测试
    for e in range(epoch):
        net.train()
        train_loss=0
        train_correct=0
        for i,(batchData,batchLabel) in enumerate(trainDataLoader):
            if use_gpu:
                batchData,batchLabel=batchData.cuda(),batchLabel.cuda()
            output=net(batchData)
            loss=criterion(output,batchLabel)
            pred=torch.argmax(output,dim=1)
            correct=torch.eq(pred,batchLabel).sum().float().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()*batchLabel.size(0)
            train_correct+=correct
            if i%50==0:
                print('epoch:{},index:{},loss:{}'.format(e+1,i+1,loss.item()))
        train_loss/=len(trainData.label)
        train_correct/=len(trainData.label)

        net.eval()
        val_loss=0
        val_correct=0
        for i, (batchData, batchLabel) in enumerate(valDataLoader):
            if use_gpu:
                batchData, batchLabel = batchData.cuda(), batchLabel.cuda()
            output = net(batchData)
            loss = criterion(output, batchLabel)
            pred = torch.argmax(output, dim=1)
            correct = torch.eq(pred, batchLabel).sum().float().item()

            val_loss += loss.item() * batchLabel.size(0)
            val_correct += correct
        val_loss/=len(valData.label)
        val_correct/=len(valData.label)

        print('*****epoch:{},train loss:{:.6f},train acc:{:.4f},val loss:{:.6f},val_acc:{:.4f}'.format(e+1,
                                                                                                       train_loss,train_correct,
                                                                                                       val_loss,val_correct))


