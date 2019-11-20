import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from collections import OrderedDict

class denseLayer(nn.Sequential):
    def __init__(self,input_features,growth_rate, bn_size, drop_rate):
        super(denseLayer, self).__init__()
        self.add_module('norm1',nn.BatchNorm2d(input_features))
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(in_channels=input_features,out_channels=bn_size*growth_rate,
                                          kernel_size=1,stride=1,bias=False))

        self.add_module('norm2',nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv2d(in_channels=bn_size*growth_rate,out_channels=growth_rate,
                                          kernel_size=3,stride=1,padding=1,bias=False))

        self.dropout=drop_rate

    def forward(self,x):
        new_feature=super(denseLayer, self).forward(x)
        if self.dropout>0:
            F.dropout(new_feature,p=self.dropout,training=self.training)
        return torch.cat((x,new_feature),1)

class denseBlock(nn.Sequential):
    def __init__(self,growth_rate,num_layers,num_input_features, bn_size, drop_rate):
        super(denseBlock, self).__init__()
        for i in range(num_layers):
            layer=denseLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module('layer%d'%(i+1),layer)

#过渡层 位于两个denseblock之间
class transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        self.features=nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features=num_init_features
        for i,num_layers in enumerate(block_config):
            block=denseBlock(growth_rate,num_layers,num_features,bn_size,drop_rate)
            self.features.add_module('denseblock%d'%(i+1),block)
            num_features=num_features+num_layers*growth_rate

            if i!=len(block_config)-1:
                trans=transition(num_features,num_features//2)
                self.features.add_module('transition%d'%(i+1),trans)
                num_features=num_features//2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool5', nn.AvgPool2d(kernel_size=7, stride=1))

        self.fc=nn.Linear(num_features,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

net=DenseNet()
net.cuda()
print(net)
result=summary(net,input_size=(3,224,224),batch_size=1)
print(result)