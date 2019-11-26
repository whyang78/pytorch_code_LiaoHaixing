import torch
from torch import nn,optim
from torchvision import models,transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from visdom import Visdom

torch.manual_seed(78)

batchsize=32
epoch=20
use_gpu=torch.cuda.is_available()
train_path='../dataset/cat_vs_dog/data/train'
val_path='../dataset/cat_vs_dog/data/val'

vis=Visdom()
vis.line([[0,0]],[0],win='loss',opts=dict(title='loss',legend=['train loss','val loss']))
vis.line([[0,0]],[0],win='accuracy',opts=dict(title='accuracy',legend=['train accuracy','val accuracy']))

#加载数据集
trainData=ImageFolder(root=train_path,transform=transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]))
trainDataLoader=DataLoader(trainData,batch_size=batchsize,shuffle=True)

valData=ImageFolder(root=val_path,transform=transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]))
valDataLoader=DataLoader(valData,batch_size=batchsize,shuffle=False)
num_classes=len(trainData.classes)

#迁移学习
fix_param=False
transfer_model=models.resnet18(pretrained=True)
if fix_param:
    for param in transfer_model.parameters():
        param.requires_grad=False
dim_in=transfer_model.fc.in_features
transfer_model.fc=nn.Linear(dim_in,num_classes)
if use_gpu:
    transfer_model.cuda()

if fix_param:
    optimizer=optim.Adam(transfer_model.fc.parameters(),lr=0.001)
else:
    #全连接层和其它层设置不同的学习率
    fc_param=transfer_model.fc.parameters()
    loc_fc_param=list(map(id,fc_param))
    base_param=list(filter(lambda p:id(p) not in loc_fc_param,transfer_model.parameters()))
    optimizer=optim.Adam([ {'params':base_param},
                           {'params':fc_param,'lr':0.005} ] , lr=0.001)

criterion=nn.CrossEntropyLoss()
if use_gpu:
    criterion=criterion.cuda()

#训练测试
for e in range(epoch):
    transfer_model.train()
    train_loss=0.0
    train_correct=0.0
    for index,(batchData,batchLabel) in enumerate(trainDataLoader):
        if use_gpu:
            batchData,batchLabel=batchData.cuda(),batchLabel.cuda()
        output=transfer_model(batchData)
        loss=criterion(output,batchLabel)
        pred=torch.argmax(output,dim=1)
        correct=torch.eq(pred,batchLabel).sum().float().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()*batchData.size(0)
        train_correct+=correct

        if (index+1)%50==0:
            print('epoch:{},index:{},loss:{}'.format(e+1,index+1,loss.item()))

    train_loss/=len(trainData.targets)
    train_correct/=len(trainData.targets)

    transfer_model.eval()
    test_loss = 0.0
    test_correct = 0.0
    with torch.no_grad():
        for batchData, batchLabel in valDataLoader:
            if use_gpu:
                batchData, batchLabel = batchData.cuda(), batchLabel.cuda()
            output = transfer_model(batchData)
            loss = criterion(output, batchLabel)
            pred = torch.argmax(output, dim=1)
            correct = torch.eq(pred, batchLabel).sum().float().item()

            test_loss += loss.item() * batchData.size(0)
            test_correct += correct
    test_loss/=len(valData.targets)
    test_correct/=len(valData.targets)

    vis.line([[train_loss,test_loss]],[e+1],win='loss',update='append')
    vis.line([[train_correct,test_correct]],[e+1],win='accuracy',update='append')
