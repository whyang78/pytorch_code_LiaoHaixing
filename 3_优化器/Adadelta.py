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
#自定义Adadelta优化器
sqrs=[]
deltas=[]
for param in net.parameters():
    sqr=torch.zeros_like(param)
    delta=torch.zeros_like(param)
    if cuda:
        sqr=sqr.cuda()
        delta=delta.cuda()
    sqrs.append(sqr)
    deltas.append(delta)
def Adadelta_optimizer(parameters,sqrs,deltas,alpha):
    '''
    :param parameters:更新参数
    :param sqrs: 梯度平方累加
    :param deltas: delta
    :param aplha: 滑动系数，衰减系数
    '''
    eps=1e-10
    for param,sqr,delta in zip(parameters,sqrs,deltas):
        sqr[:]=alpha*sqr+(1-alpha)*param.grad.data**2
        current_grad=torch.sqrt(delta+eps)/torch.sqrt(sqr+eps)*param.grad.data
        delta[:]=alpha*delta+(1-alpha)*current_grad**2
        param.data=param.data-current_grad
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
        Adadelta_optimizer(net.parameters(),sqrs,deltas,0.9)

        viz.line([loss.item()],[global_step+1],win='loss',update='append')
        global_step+=1


