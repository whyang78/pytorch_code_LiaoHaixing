import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import save_image
from matplotlib import  pyplot as plt
from PIL import Image
import os
import glob

save_path='./results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.e_conv1=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1,stride=1,padding=0,bias=True)
        self.e_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)
        self.e_conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3, bias=True)
        self.e_conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        sources=[]
        sources.append(x)

        x1=self.relu(self.e_conv1(x))
        x2=self.relu(self.e_conv2(x1))
        concat1=torch.cat((x1,x2),dim=1)
        x3=self.relu(self.e_conv3(concat1))
        concat2=torch.cat((x2,x3),dim=1)
        x4=self.relu(self.e_conv4(concat2))
        concat3=torch.cat((x1,x2,x3,x4),dim=1)
        x5=self.relu(self.e_conv5(concat3))
        clean=self.relu( (x5 * x) - x5 + 1 )
        return clean

class testdataset(Dataset):
    def __init__(self,root,transform=None):
        self.root=root
        self.transfrom=transform
        self.path=sorted(glob.glob(os.path.join(self.root, '*')))

    def __getitem__(self, index):
        path=self.path[index]
        data=path
        if transforms is not None:
            data=self.transfrom(data)
        return data,path

    def __len__(self):
        return len(self.path)


tf=transforms.Compose([
    transforms.Lambda(lambda x:Image.open(x)),
    transforms.ToTensor()
])
datapath='./test_images'
testdata=testdataset(datapath,transform=tf)
testdataloader=DataLoader(testdata,batch_size=1,shuffle=False)

net=AODNet()
weight_path='./dehazer.pth'
weight=torch.load(weight_path)
net.load_state_dict(weight)

net.eval()
for data,path in testdataloader:
    clean=net(data)
    path_=os.path.join(save_path,str(path[0]).split(os.sep)[-1])
    save_image(torch.cat((data,clean),dim=0),path_)
    print('path:',path_,', DONE!')





