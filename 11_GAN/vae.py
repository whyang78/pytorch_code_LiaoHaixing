import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

bs=128
image_size=784
hidden_dim=400
z_dim=20
lr=0.001
epoch=20

save_result='./vae_result'
if not os.path.exists(save_result):
    os.makedirs(save_result)

data=datasets.MNIST('./data',train=True,transform=transforms.ToTensor(),download=False)
dataloader=DataLoader(data,batch_size=bs,shuffle=False)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1=nn.Linear(image_size,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,z_dim)
        self.fc3=nn.Linear(hidden_dim,z_dim)
        self.fc4=nn.Linear(z_dim,hidden_dim)
        self.fc5=nn.Linear(hidden_dim,image_size)

    def encode(self,x):
        h=F.relu(self.fc1(x))
        mu=self.fc2(h)
        log_var=self.fc3(h)
        return mu,log_var

    def reparameterize(self,mu,log_var):
        std=torch.exp(log_var/2)
        eps=torch.randn_like(std)
        return mu + eps * std

    def decode(self,x):
        h=F.relu(self.fc4(x))
        rec=F.sigmoid(self.fc5(h))
        return rec

    def forward(self, x):
        mu,log_var=self.encode(x)
        z=self.reparameterize(mu,log_var)
        rec=self.decode(z)
        return rec,mu,log_var

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net=VAE().to(device)
optimizer=optim.Adam(net.parameters(),lr=lr)

for e in range(epoch):
    net.train()
    for i,(batchdata,_) in enumerate(dataloader):
        batchdata=batchdata.to(device).view(-1,image_size)
        rec,mu,log_var=net(batchdata)

        rec_loss=F.binary_cross_entropy(rec,batchdata,size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss= rec_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%50==0:
            print('epoch:{},index:{},loss:{:.4f},rec_loss:{:.4f},kl_loss:{:.4f}'.format(e+1,i+1,
                                                                                        loss.item(),rec_loss.item(),
                                                                                        kl_div.item()))

    with torch.no_grad():
        net.eval()
        # 采样
        z=torch.randn(bs,z_dim).to(device)
        dz=net.decode(z).view(-1,1,28,28).cpu()
        save_image(dz,os.path.join(save_result,'sample_{}.png'.format(e+1)))
        #重构
        rec, _, _ = net(batchdata)
        image_concat=torch.cat((batchdata.view(-1,1,28,28).cpu(),rec.view(-1,1,28,28).cpu()),dim=3)
        save_image(image_concat,os.path.join(save_result,'rec_{}.png'.format(e+1)))






