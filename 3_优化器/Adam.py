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
#自定义Adam优化器
t=1
sqrs=[]
vs=[]
for param in net.parameters():
    sqr=torch.zeros_like(param)
    v=torch.zeros_like(param)
    if cuda:
        sqr=sqr.cuda()
        v=v.cuda()
    sqrs.append(sqr)
    vs.append(v)
def Adam_optimizer(parameters,sqrs,vs,t,lr,beta1=0.9,beta2=0.999):
    '''
    :param parameters:更新参数
    :param sqrs: 梯度平方累加
    :param vs: 历史梯度
    :param t: 更新次数
    :param beta1: 一阶矩系数
    :param beta2:二阶矩系数
    '''
    eps=1e-8
    for param,sqr,v in zip(parameters,sqrs,vs):
        v[:]=beta1*v+(1-beta1)*param.grad.data
        sqr[:]=beta2*sqr+(1-beta2)*param.grad.data**2
        v_hat=v/(1-beta1**t)
        sqr_hat=sqr/(1-beta2**t)
        param.data=param.data-lr*v_hat/torch.sqrt(sqr_hat+eps)
# optimizer=optim.Adagrad(net.parameters(),lr=lr)
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
        Adam_optimizer(net.parameters(),sqrs,vs,t,0.001)
        t+=1

        viz.line([loss.item()],[global_step+1],win='loss',update='append')
        global_step+=1


