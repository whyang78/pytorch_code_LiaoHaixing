import os
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from visdom import Visdom

#超参数
cuda=torch.cuda.is_available()
bs=32
lr=0.01
epoch=2
dowmload=False
dataset_path='../dataset/mnist'

if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
    os.makedirs(dataset_path)
    dowmload=True

#加载数据集
trainData=datasets.MNIST(dataset_path,train=True,transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize([0.5],[0.5])
]),download=dowmload)
trainDataLoader=DataLoader(trainData,batch_size=bs,shuffle=True)

testData=datasets.MNIST(dataset_path,train=False,transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize([0.5],[0.5])
]),download=dowmload)
testDataLoader=DataLoader(testData,batch_size=bs,shuffle=False)

#构建MLP
net=nn.Sequential(nn.Linear(28*28,256),
                  nn.ReLU(True),
                  nn.Linear(256,64),
                  nn.ReLU(True),
                  nn.Linear(64,10))
#自定义Momentum优化器
vs=[]
for param in net.parameters():
    v=torch.zeros_like(param)
    if cuda:
        v=v.cuda()
    vs.append(v)
def Momentum_optimizer(parameters,vs,lr,momentum):
    '''
    :param parameters:更新参量
    :param vs: 以往梯度
    :param lr: 学习率
    :param momentum: 动量系数
    '''
    for param,v in zip(parameters,vs):
        v[:]=momentum*v+lr*param.grad.data
        param.data=param.data-v
# optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9)
criterion=nn.CrossEntropyLoss()

if cuda:
    net.cuda(0)
    criterion.cuda(0)

#使用visdom绘图
viz=Visdom()
viz.line([0.0],[0.0],win='loss',opts=dict(title='train loss',legend=['train loss']))

global_step=0
for e in range(epoch):
    net.train()
    for batchData,batchLabel in trainDataLoader:
        batchData=batchData.view(-1,28*28)
        if cuda:
            batchData,batchLabel=batchData.cuda(0),batchLabel.cuda(0)
        output=net(batchData)
        loss=criterion(output,batchLabel)

        net.zero_grad()
        loss.backward()
        Momentum_optimizer(net.parameters(),vs,lr,0.9)

        viz.line([loss.item()],[global_step+1],win='loss',update='append')
        global_step+=1


