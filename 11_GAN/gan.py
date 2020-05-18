import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

bs=100
image_size=784
hidden_dim=256
noise_dim=64
lr=0.0002
epoch=200

save_result='./gan_result'
if not os.path.exists(save_result):
    os.makedirs(save_result)

tf=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
data=datasets.MNIST('./data',train=True,transform=tf,download=False) # [-1,1]
dataloader=DataLoader(data,batch_size=bs,shuffle=False)

D=nn.Sequential(
    nn.Linear(image_size,hidden_dim),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim,hidden_dim),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim,1),
    nn.Sigmoid()
)
G=nn.Sequential(
    nn.Linear(noise_dim,hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim,image_size),
    nn.Tanh()
)

device=torch.device('cuda:0')
D=D.to(device)
G=G.to(device)
D_optimizer=optim.Adam(D.parameters(),lr=lr)
G_optimizer=optim.Adam(G.parameters(),lr=lr)
criterion=nn.BCELoss().to(device)

for e in range(epoch):
    for i,(batchdata,_) in enumerate(dataloader):
        batch=batchdata.size(0)
        batchdata=batchdata.to(device).view(batch,-1)
        real_labels=torch.ones(batch,1).to(device)
        fake_labels=torch.zeros(batch,1).to(device)
        #训练判别器
        real_score=D(batchdata)
        real_loss=criterion(real_score,real_labels)

        noise=torch.randn(batch,noise_dim).to(device)
        fake_data=G(noise).detach() # 防止生成器参数更新
        fake_score=D(fake_data)
        fake_loss=criterion(fake_score,fake_labels)

        d_loss=real_loss+fake_loss
        D_optimizer.zero_grad()
        d_loss.backward()
        D_optimizer.step()

        #训练生成器
        noise = torch.randn(batch, noise_dim).to(device)
        fake_data=G(noise)
        score=D(fake_data)
        g_loss=criterion(score,real_labels)
        G_optimizer.zero_grad()
        g_loss.backward()
        G_optimizer.step()

        if(i+1)%50==0:
            print('epoch:{},index:{},g_loss:{:.4f},d_loss:{:.4f},d(g)_score:{:.4f},d_score:{:.4f}'.format(e+1,i+1,
                                                                                        g_loss.item(),d_loss.item(),
                                                                                        fake_score.mean().item(),
                                                                                        real_score.mean().item()))

    fake_images=fake_data.view(batch,-1,28,28).cpu()
    fake_images=(fake_images * 0.5 + 0.5).clamp(0,1)
    save_image(fake_images,os.path.join(save_result,'gfake_{}.png'.format(e+1)))

