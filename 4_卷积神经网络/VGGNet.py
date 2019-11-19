import torchvision
from torch import nn
from torchsummary import summary

class VGGnet(nn.Module):
    def __init__(self,feature,num_classes=1000,init_weights=False):
        super(VGGnet, self).__init__()
        self.feature=feature
        self.avgpool=nn.AdaptiveAvgPool2d((7,7))
        self.classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x=self.feature(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg,in_channel,bn=False):
    layers=[]
    inchannel=in_channel
    for c in cfg:
        if c=='M':
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv=nn.Conv2d(inchannel,c,kernel_size=3,stride=1,padding=1)
            if bn:
                layers+=[conv,nn.BatchNorm2d(c),nn.ReLU(inplace=True)]
            else:
                layers+=[conv,nn.ReLU(inplace=True)]
            inchannel=c
    return nn.Sequential(*layers)


vgg16=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
net=VGGnet(make_layers(vgg16,in_channel=3),init_weights=True)
net.cuda()
print(net)
result=summary(net,input_size=(3,224,224),batch_size=1)
print(result)