import torchvision
from torch import nn
from torchsummary import summary

class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(9216,4096,bias=True),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(4096,1000,bias=True)
        )

    def forward(self, x):
        x=self.layer1(x)
        x=x.view(-1,9216)
        x=self.layer2(x)
        return x


# net=torchvision.models.AlexNet()
net=Alexnet()
net.cuda()
print(net)
result=summary(net,input_size=(3,224,224),batch_size=1)
print(result)