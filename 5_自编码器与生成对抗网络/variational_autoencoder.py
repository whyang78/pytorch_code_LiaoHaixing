import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image

torch.manual_seed(78)

lr=0.001
batchsize=32
epoch=20
dataset_path='../dataset/mnist'

#加载数据集 只使用训练集就可以
trainData=datasets.MNIST(root=dataset_path,train=True,transform=transforms.ToTensor(),download=False)
trainDataLoader=DataLoader(trainData,batch_size=batchsize,shuffle=True)

#构建网络
class vae(nn.Module):
    '''
    全连接层搭建变分自动编码器
    '''
    def __init__(self,in_dim,hidden_dim):
        super(vae, self).__init__()

        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,64)
        self.fc31=nn.Linear(64,16)
        self.fc32=nn.Linear(64,16)
        self.fc4=nn.Linear(16,64)
        self.fc5=nn.Linear(64,256)
        self.fc6=nn.Linear(256,28*28)

    def encoder(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mu=self.fc31(x)
        logvar=self.fc32(x)
        return mu,logvar

    def sample(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        s=std*torch.randn_like(std)+mu
        return s

    def decoder(self,x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x

    def forward(self, x):
        mu,logvar=self.encoder(x)
        sample=self.sample(mu,logvar)
        decode=self.decoder(sample)
        return decode,mu,logvar

device=torch.device('cuda:0')
net=vae().to(device)
optimizer=optim.Adam(net.parameters(),lr=lr)

def loss_function(output,target,mu,logvar):
    rebuild_loss=F.mse_loss(output,target,size_average=False)
    #loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss=-0.5*torch.sum(logvar+1-mu.pow(2)-logvar.exp())
    return rebuild_loss+kl_loss

for e in range(epoch):
    for i,(datas,_) in enumerate(trainDataLoader):
        datas=datas.view(datas.size(0),-1)
        datas=datas.to(device)
        output,mu,logvar=net(datas)
        loss=loss_function(output,datas,mu,logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%50==0:
            print('epoch:{},index:{},loss:{}'.format(e+1,i+1,loss.item()))

    pic=output.view(output.size(0),1,28,28)
    save_image(pic.cpu().data,'./sample_autoencoder/image_{}.png'.format(e+1))


