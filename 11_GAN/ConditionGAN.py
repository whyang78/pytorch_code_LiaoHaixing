import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image,make_grid
from matplotlib import pyplot as plt
import os

bs=100
lr=0.0001
epoch=50

save_result='./cgan_result'
if not os.path.exists(save_result):
    os.makedirs(save_result)

tf=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
data=datasets.MNIST('./data',train=True,transform=tf,download=False) # [-1,1]
dataloader=DataLoader(data,batch_size=bs,shuffle=False)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat((x, c), 1)
        out = self.model(x)
        return out.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat((z, c), 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

device=torch.device('cpu')
D=Discriminator().to(device)
G=Generator().to(device)
criterion=nn.BCELoss().to(device)
D_optimizer=optim.Adam(D.parameters(),lr=lr)
G_optimizer=optim.Adam(G.parameters(),lr=lr)

for e in range(epoch):
    D.train()
    G.train()
    for i,(batchdata,batchlabel) in enumerate(dataloader):
        batch=batchdata.size(0)
        batchdata,batchlabel=batchdata.to(device),batchlabel.to(device)
        real=torch.ones(batch).to(device)
        fake=torch.zeros(batch).to(device)
        #训练判别器
        real_score=D(batchdata,batchlabel)
        real_loss=criterion(real_score,real)

        noise=torch.randn(batch,100).to(device)
        fake_labels=torch.randint(0,10,(batch,)).to(device)
        fake_data=G(noise,fake_labels).detach()
        fake_score=D(fake_data,fake_labels)
        fake_loss=criterion(fake_score,fake)

        d_loss=real_loss+fake_loss
        D_optimizer.zero_grad()
        d_loss.backward()
        D_optimizer.step()

        #训练生成器
        noise = torch.randn(batch, 100).to(device)
        fake_labels = torch.randint(0, 10, (batch,)).to(device)
        fake_data = G(noise, fake_labels)
        score = D(fake_data, fake_labels)
        g_loss = criterion(score, real)

        G_optimizer.zero_grad()
        g_loss.backward()
        G_optimizer.step()

        if (i+1)%50==0:
            print('epoch:{},index:{},g_loss:{:.4f},d_loss:{:.4f},d(g)_score:{:.4f},d_score:{:.4f}'.format(e + 1, i + 1,
                                                                                                          g_loss.item(),
                                                                                                          d_loss.item(),
                                                                                                          fake_score.mean().item(),
                                                                                                          real_score.mean().item()))
    i=torch.randint(0,batch,(1,))
    fake_images = fake_data[int(i[0])].view(1, -1, 28, 28).cpu()
    fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)
    save_image(fake_images, os.path.join(save_result, 'gfake_{}_{}.png'.format(e + 1,fake_labels[int(i[0])])))

    torch.save(G.state_dict(), './G.ckpt')
    torch.save(D.state_dict(), './D.ckpt')

# 绘制10*10
def plot_1(G):
    noise = torch.randn(100, 100).to(device)
    labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)
    fake_image = G(noise, labels).unsqueeze(1)
    fake_image=fake_image * 0.5 + 0.5
    grid=make_grid(fake_image,nrow=10,normalize=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='binary')
    ax.axis('off')
    plt.show()

#绘制指定数字的图像
def plot_2(G,digital):
    noise = torch.randn(1, 100).to(device)
    labels = torch.LongTensor([digital]).to(device)
    fake_image = G(noise, labels)
    fake_image=transforms.ToPILImage()(fake_image)

    # plt.figure()
    plt.imshow(fake_image)
    plt.title(str(labels.numpy()))
    plt.axis('off')
    # plt.show()

G_weight=torch.load('./G.ckpt',device)
G.load_state_dict(G_weight)
plot_1(G)
plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plot_2(G,i)
plt.show()



