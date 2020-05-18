import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

data_path='./data'
epoch=10
train_bs=128
test_bs=1
lr=0.001

tf=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768])
])
train_data=datasets.CIFAR10(root=data_path,train=True,transform=tf,download=False)
train_dataloader=DataLoader(train_data,batch_size=train_bs,shuffle=True)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=x.view(-1,36*6*6)
        x=F.relu(self.fc2(F.relu(self.fc1(x))))
        return x

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net=CNNNet().to(device)
criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9)

for e in range(epoch):
    net.train()
    total_loss = 0.0
    for batchdata,batchlabel in train_dataloader:
        batchdata,batchlabel=batchdata.to(device),batchlabel.to(device)
        output=net(batchdata)
        loss=criterion(output,batchlabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.cpu().item()*len(batchdata)
    total_loss/=len(train_dataloader.dataset)
    print('epoch:{},loss:{:.4f}'.format(e,total_loss))


#拿训练完的模型进行特征可视化
test_data=datasets.CIFAR10(root=data_path,train=False,transform=tf,download=False)
test_dataloader=DataLoader(test_data,batch_size=test_bs,shuffle=False)
img,label=iter(test_dataloader).next()

img_grid=make_grid(img,normalize=True,scale_each=True)
plt.imshow(img_grid.numpy().transpose((1,2,0)))
plt.show()

net.to(torch.device('cpu'))
x=img
net.eval()
for name,layer in net._modules.items():
    x=x.view(x.size(0),-1) if 'fc' in name else x
    x=layer(x)
    x=F.relu(x) if 'conv' in name or 'fc' in name else x

    if 'conv' in name:
        print(f'{name}')
        x_=x.transpose(0,1)  # 将卷积核个数提前，每张特征图对应一个卷积核得到的特征图
        feature_grid=make_grid(x_,normalize=True,scale_each=True,nrow=4)
        plt.imshow(feature_grid.detach().numpy().transpose((1, 2, 0)))
        plt.show()
