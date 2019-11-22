import torch
from torch import nn,optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,sampler
from torch.nn import functional as F
from visdom import Visdom

viz=Visdom()
viz.line([[0,0]],[0],win='loss',opts=dict(title='loss',legend=['g_loss','d_loss']))

batchsize=32
epoch=100
noise_dim=100
dataset_path='../dataset/mnist'

class chunkSampler(sampler.Sampler):
    def __init__(self,numSamplers,start=0):
        self.numSamplers=numSamplers
        self.start=start

    def __iter__(self):
        return iter(range(self.start,self.start+self.numSamplers))

    def __len__(self):
        return self.numSamplers

#加载数据集 自定义sampler，只取前5000张训练集数据进行训练
numTrain=5000
trainData=datasets.MNIST(root=dataset_path,train=True,transform=transforms.ToTensor(),download=False)
# trainDataLoader=DataLoader(trainData,batch_size=batchsize,sampler=chunkSampler(numTrain,0))
trainDataLoader=DataLoader(trainData,batch_size=batchsize,shuffle=True)

#构建网络
#判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

#生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)  # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x

#损失函数
def d_loss(logits_real,logits_fake):
    real_labels=torch.ones((logits_real.size(0),1),dtype=torch.float32,device=logits_real.device)
    fake_labels=torch.zeros((logits_fake.size(0),1),dtype=torch.float32,device=logits_fake.device)
    loss=F.binary_cross_entropy_with_logits(logits_real,real_labels)+\
         F.binary_cross_entropy_with_logits(logits_fake,fake_labels)
    return loss

def g_loss(logits_fake):
    real_labels=torch.ones((logits_fake.size(0),1),dtype=torch.float32,device=logits_fake.device)
    loss=F.binary_cross_entropy_with_logits(logits_fake,real_labels)
    return loss

device=torch.device('cuda:0')
d_net=discriminator().to(device)
g_net=generator().to(device)

d_optimizer=optim.Adam(d_net.parameters(),lr=5e-4,betas=(0.5,0.999))
g_optimizer=optim.Adam(g_net.parameters(),lr=5e-4,betas=(0.5,0.999))

#训练过程
iternum=0
for e in range(epoch):
    for i, (datas, _) in enumerate(trainDataLoader):
        #训练判别器
        datas = datas.to(device)
        logits_real=d_net(datas)

        sample_noise=torch.rand((datas.size(0),noise_dim),dtype=torch.float32,device=device)
        fake=g_net(sample_noise).detach()
        logits_fake=d_net(fake)

        loss_d=d_loss(logits_real,logits_fake)

        d_optimizer.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        #训练生成器
        sample_noise = torch.rand((batchsize, noise_dim), dtype=torch.float32, device=device)
        fake = g_net(sample_noise)
        logits_fake = d_net(fake)
        loss_g=g_loss(logits_fake)

        g_optimizer.zero_grad()
        loss_g.backward()
        g_optimizer.step()

        if (iternum+1)%50==0:
            viz.line([[loss_g.item(),loss_d.item()]],[iternum+1],win='loss',update='append')
            viz.images(fake.view(-1,1,28,28).cpu(),nrow=8,win='images')
        iternum+=1