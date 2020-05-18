import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import numpy as np
import random
from collections import Counter
from mynet import CNNNet,AVPNet,LeNet

torch.manual_seed(78)
random.seed(78)
np.random.seed(78)

download=False
bs=128
lr=0.001
epoch=10
datapath='./data'

train_tf=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768])
])
test_tf=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768])
])
train_data=datasets.CIFAR10(datapath,train=True,transform=train_tf,download=download)
train_dataloader=DataLoader(train_data,batch_size=bs,shuffle=True)
test_data=datasets.CIFAR10(datapath,train=False,transform=test_tf,download=download)
test_dataloader=DataLoader(test_data,batch_size=bs,shuffle=False)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net1=CNNNet().to(device)
net2=AVPNet().to(device)
net3=LeNet().to(device)
nets=[net1,net2,net3]
optimizer=optim.Adam([{'params':net.parameters()} for net in nets],lr=lr)
criterion=nn.CrossEntropyLoss().to(device)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#************************************多模型投票*******************************#
for e in range(epoch):
    for batchdata,batchlabel in train_dataloader:
        batchdata,batchlabel=batchdata.to(device),batchlabel.to(device)
        optimizer.zero_grad()
        for net in nets:
            net.train()
            output=net(batchdata)
            loss=criterion(output,batchlabel)
            loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred=[]
        vote_correct=0.0
        nets_correct=[0.0 for i in range(len(nets))]
        classes_correct=[0.0 for i in range(len(classes))]
        classes_total=[0.0 for i in range(len(classes))]
        for batchdata,batchlabel in test_dataloader:
            batchdata,batchlabel=batchdata.to(device),batchlabel.to(device)
            for i,net in enumerate(nets):
                net.eval()
                output=net(batchdata)
                prediction=torch.argmax(output,dim=1)
                correct=torch.eq(prediction,batchlabel).sum().cpu().item()
                nets_correct[i]+=correct
                pred.append(prediction.cpu().numpy())

            all_pred=np.array(pred)
            pred.clear()
            results=[Counter(all_pred[:,i]).most_common(1)[0][0] for i in range(len(batchlabel))]
            correct=(np.array(results)==batchlabel.cpu().numpy()).sum()
            vote_correct+=correct

            c=(np.array(results)==batchlabel.cpu().numpy())
            for i in range(len(batchlabel)):
                label=batchlabel[i]
                classes_correct[label]+=c[i]
                classes_total[label]+=1
        print("epoch:" + str(e) + "集成模型的正确率" + str(vote_correct / len(test_dataloader)))
        for idx, correct in enumerate(nets_correct):
            print("模型" + str(idx) + "的正确率为：" + str(correct / len(test_dataloader)))
        for i in range(len(classes)):
            print('类别'+str(classes[i])+'的正确率为：'+ str(classes_correct[i]/classes_total[i]))


#*******************************多模型带权投票***********************************#
for e in range(epoch):
    for batchdata,batchlabel in train_dataloader:
        batchdata,batchlabel=batchdata.to(device),batchlabel.to(device)
        optimizer.zero_grad()
        for net in nets:
            net.train()
            output=net(batchdata)
            loss=criterion(output,batchlabel)
            loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred=[]
        vote_correct=0.0
        nets_correct=[0.0 for i in range(len(nets))]
        classes_correct=[0.0 for i in range(len(classes))]
        classes_total=[0.0 for i in range(len(classes))]
        for batchdata,batchlabel in test_dataloader:
            batchdata,batchlabel=batchdata.to(device),batchlabel.to(device)
            for i,net in enumerate(nets):
                net.eval()
                output=net(batchdata)
                prediction=torch.argmax(output,dim=1)
                correct=torch.eq(prediction,batchlabel).sum().cpu().item()
                nets_correct[i]+=correct
                pred.append(prediction.cpu().numpy())

            all_pred=np.array(pred)
            pred.clear()
            net_weight = np.array([nets_correct[i] / sum(nets_correct) for i in range(len(nets_correct))])
            results=[]
            for i in range(len(batchlabel)):
                sample_pred=all_pred[:,i]
                sample_classes=list(set(sample_pred))
                sample_classes_num=[0.0 for i in range(len(sample_classes))]
                for j,sc in enumerate(sample_classes):
                    mask=(sample_pred==sc).astype('int')
                    sample_classes_num[j]=np.dot(net_weight,mask)
                results.append(sample_classes[np.argmax(sample_classes_num)])

            correct=(np.array(results)==batchlabel.cpu().numpy()).sum()
            vote_correct+=correct

            c=(np.array(results)==batchlabel.cpu().numpy())
            for i in range(len(batchlabel)):
                label=batchlabel[i]
                classes_correct[label]+=c[i]
                classes_total[label]+=1
        print("epoch:" + str(e) + "集成模型的正确率" + str(vote_correct / len(test_dataloader)))
        for idx, correct in enumerate(nets_correct):
            print("模型" + str(idx) + "的正确率为：" + str(correct / len(test_dataloader)))
        for i in range(len(classes)):
            print('类别'+str(classes[i])+'的正确率为：'+ str(classes_correct[i]/classes_total[i]))


