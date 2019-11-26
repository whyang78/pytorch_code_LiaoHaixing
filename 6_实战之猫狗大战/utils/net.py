import torch
from torch import nn
from torchvision import models

class feature_net(nn.Module):
    def __init__(self,model):
        '''
        :param model:vgg16 / inceptionv3 / resnet18
        '''
        super(feature_net, self).__init__()

        if model=='vgg16':
            net=models.vgg16(pretrained=True)
            self.feature=nn.Sequential(*list(net.children())[:-2])
            self.feature.add_module('global average', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        elif model=='inceptionv3':
            net=models.inception_v3(pretrained=True)
            self.feature=nn.Sequential(*list(net.children())[:-1])
            self.feature._modules.pop('13')  #此处是._modules(有一个横杠)
            self.feature.add_module('global average', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        elif model=='resnet18':
            net=models.resnet18(pretrained=True)
            self.feature = nn.Sequential(*list(net.children())[:-1])

    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        return x


class classifier(nn.Module):
    def __init__(self,feature_dim,numclasses):
        super(classifier, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(feature_dim,1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000,numclasses)
        )

    def forward(self,x):
        x=self.fc(x)
        return x


