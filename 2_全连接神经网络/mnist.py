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
epoch=20
dowmload=False
dataset_path='./mnist'

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
optimizer=optim.SGD(net.parameters(),lr=lr)
criterion=nn.CrossEntropyLoss()

if cuda:
    net.cuda(0)
    criterion.cuda(0)

#使用visdom绘图
viz=Visdom()
viz.line([[0.0,0.0]],[0.0],win='loss',opts=dict(title='train & test loss',legend=['train loss','test loss']))
viz.line([[0.0,0.0]],[0.0],win='accuracy',opts=dict(title='train & test accuracy',legend=['train accuracy','test accuracy']))

#训练
for e in range(epoch):
    train_loss=0.0
    train_correct=0.0
    net.train()
    for batchData,batchLabel in trainDataLoader:
        batchData=batchData.view(-1,28*28)
        if cuda:
            batchData,batchLabel=batchData.cuda(0),batchLabel.cuda(0)
        output=net(batchData)
        loss=criterion(output,batchLabel)
        pred=torch.argmax(output,dim=1)
        correct=torch.eq(pred,batchLabel).sum().float().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        train_correct+=correct
    train_loss/=len(trainDataLoader.dataset)
    train_correct/=len(trainDataLoader.dataset)

    #测试
    val_loss = 0.0
    val_correct = 0.0
    net.eval()
    for batchData, batchLabel in testDataLoader:
        batchData=batchData.view(-1, 28 * 28)
        if cuda:
            batchData, batchLabel = batchData.cuda(0), batchLabel.cuda(0)
        output = net(batchData)
        loss = criterion(output,batchLabel)
        pred = torch.argmax(output, dim=1)
        correct = torch.eq(pred, batchLabel).sum().float().item()

        val_loss += loss.item()
        val_correct += correct
    val_loss /= len(testDataLoader.dataset)
    val_correct /= len(testDataLoader.dataset)

    viz.line([[train_loss,val_loss]],[e+1],win='loss',update='append')
    viz.line([[train_correct, val_correct]], [e+1], win='accuracy',update='append')
    print('epoch:{},train accuracy:{:.4f},train loss:{:.4f},test accuracy:{:.4f},test loss:{:.4f}'
          .format(e+1,train_correct,train_loss,val_correct,val_loss))

