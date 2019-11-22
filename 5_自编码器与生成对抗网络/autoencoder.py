import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

lr=0.001
batchsize=32
epoch=20
dataset_path='../dataset/mnist'

#加载数据集 只使用训练集就可以
trainData=datasets.MNIST(root=dataset_path,train=True,transform=transforms.ToTensor(),download=False)
trainDataLoader=DataLoader(trainData,batch_size=batchsize,shuffle=True)

#构建网络
class ae_linear(nn.Module):
    '''
    全连接层搭建自编码器
    '''
    def __init__(self,in_dim,hidden_dim):
        super(ae_linear, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 4, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class ae_conv(nn.Module):
    '''
    卷积层搭建自编码器
    '''
    def __init__(self):
        super(ae_conv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

device=torch.device('cuda:0')
# net=ae_linear(28*28,128).to(device)
net=ae_conv().to(device)
optimizer=optim.Adam(net.parameters(),lr=lr)
criterion=nn.MSELoss().to(device)

for e in range(epoch):
    for i,(datas,_) in enumerate(trainDataLoader):
        # datas=datas.view(datas.size(0),-1)  #若要使用ae_linear，则要进行view
        datas=datas.to(device)
        _,output=net(datas)
        loss=criterion(output,datas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%50==0:
            print('epoch:{},index:{},loss:{}'.format(e+1,i+1,loss.item()))

    if not os.path.exists('./sample_autoencoder'):
        os.makedirs('./sample_autoencoder')
    pic=output.view(output.size(0),1,28,28)
    save_image(pic.cpu().data,'./sample_autoencoder/image_{}.png'.format(e+1))


# #可视化编码分布  因编码是3维，故绘制三维图像
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
#
# view_data = trainData.train_data[:200].float().view(-1,28*28)/255.0
# net.cpu()
# encode, _ = net(view_data)    # 提取压缩的特征值
# fig = plt.figure(2)
# ax = Axes3D(fig)    # 3D 图
# # x, y, z 的数据值
# X = encode.data[:, 0].numpy()
# Y = encode.data[:, 1].numpy()
# Z = encode.data[:, 2].numpy()
# values = trainData.train_labels[:200].numpy()  # 标签值
# for x, y, z, s in zip(X, Y, Z, values):
#     c = cm.rainbow(int(255*s/9))    # 上色
#     ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())
# ax.set_zlim(Z.min(), Z.max())
# plt.show()