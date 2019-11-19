from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock, self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel,out_channels=outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.right=nn.Sequential()
        if stride!=1 or inchannel!=outchannel:
            self.right=nn.Sequential(
                nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        x1=self.left(x)
        x2=self.right(x)
        x=x1+x2
        x=F.relu(x)
        return x

class Resnet18(nn.Module):
    def __init__(self,numclasses=1000):
        super(Resnet18, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.inchannel=64

        self.block1=self.make_layers(ResidualBlock,64,2,1)
        self.block2=self.make_layers(ResidualBlock,128,2,2)
        self.block3 = self.make_layers(ResidualBlock, 256, 2, 2)
        self.block4 = self.make_layers(ResidualBlock, 512, 2, 2)

        self.fc=nn.Linear(512,numclasses)

    def make_layers(self,block,outchannel,num,stride):
        strides=[stride]+[1]*(num-1)
        layers=[]
        for s in strides:
            layers.append(block(self.inchannel,outchannel,s))
            self.inchannel=outchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv(x)
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=F.adaptive_avg_pool2d(x,(1,1))
        x=x.view(-1,512)
        x=self.fc(x)
        return x

if __name__ == '__main__':

    net=Resnet18()
    net.cuda()
    print(net)
    result=summary(net,input_size=(3,224,224),batch_size=1)
    print(result)