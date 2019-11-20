import torch
from torch import nn
from torchsummary import summary


class BasicConv(nn.Module):
    def __init__(self,inchannel,outchannel,**kwargs):
        super(BasicConv, self).__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,bias=False,**kwargs),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x=self.layer(x)
        return x


class  Inception(nn.Module):
    def __init__(self,inchannel,ch1x1,ch3x3pre,ch3x3,ch5x5pre,ch5x5,poolchannel):
        super(Inception, self).__init__()

        self.branch1=BasicConv(inchannel,ch1x1,kernel_size=1)

        self.branch2=nn.Sequential(
            BasicConv(inchannel,ch3x3pre,kernel_size=1),
            BasicConv(ch3x3pre,ch3x3,kernel_size=3,padding=1)
        )

        self.branch3=nn.Sequential(
            BasicConv(inchannel,ch5x5pre,kernel_size=1),
            BasicConv(ch5x5pre,ch5x5,kernel_size=5,padding=2)
        )

        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1,ceil_mode=True),
            BasicConv(inchannel,poolchannel,kernel_size=1)
        )

    def forward(self, x):
        x1=self.branch1(x)
        x2=self.branch2(x)
        x3=self.branch3(x)
        x4=self.branch4(x)
        x=torch.cat((x1,x2,x3,x4),dim=1)
        return x

class googlenet(nn.Module):
    def __init__(self,numclasses=1000):
        super(googlenet, self).__init__()

        self.conv1=BasicConv(3,64,kernel_size=7,stride=2,padding=3)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.conv2=BasicConv(64,64,kernel_size=1)
        self.conv3=BasicConv(64,192,kernel_size=3,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.inception3a=Inception(192,64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout(0.4)
        self.fc=nn.Linear(1024,numclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

net=googlenet()
net.cuda()
print(net)
result = summary(net, input_size=(3, 224, 224), batch_size=1)
print(result)