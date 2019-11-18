import torch
from torch import nn
from torchsummary import summary


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0,bias=True)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0,bias=True)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=x.view(-1,400)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

model=Lenet().cuda()
print(model)
result=summary(model,input_size=(1,32,32),batch_size=1)
print(result)