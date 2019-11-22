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
discriminator=nn.Sequential(
    nn.Linear(28*28,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,32),
    nn.LeakyReLU(0.2),
    nn.Linear(32,1)
)
#生成器
generator = nn.Sequential(
    nn.Linear(noise_dim, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 784),
    nn.Sigmoid()
    )

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
discriminator.to(device)
generator.to(device)

d_optimizer=optim.Adam(discriminator.parameters(),lr=5e-4,betas=(0.5,0.999))
g_optimizer=optim.Adam(generator.parameters(),lr=5e-4,betas=(0.5,0.999))

#训练过程
iternum=0
for e in range(epoch):
    # for j in range(5):
    for i, (datas, _) in enumerate(trainDataLoader):
        #训练判别器
        datas = datas.view(datas.size(0), -1)
        datas = datas.to(device)
        logits_real=discriminator(datas)

        sample_noise=torch.rand((datas.size(0),noise_dim),dtype=torch.float32,device=device)
        fake=generator(sample_noise).detach()
        logits_fake=discriminator(fake)

        loss_d=d_loss(logits_real,logits_fake)

        d_optimizer.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        #训练生成器
        sample_noise = torch.rand((batchsize, noise_dim), dtype=torch.float32, device=device)
        fake = generator(sample_noise)
        logits_fake = discriminator(fake)
        loss_g=g_loss(logits_fake)

        g_optimizer.zero_grad()
        loss_g.backward()
        g_optimizer.step()

        if (iternum+1)%50==0:
            viz.line([[loss_g.item(),loss_d.item()]],[iternum+1],win='loss',update='append')
            viz.images(fake.view(-1,1,28,28).cpu(),nrow=8,win='images')
        iternum+=1