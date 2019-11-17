import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn,optim

#加载数据
with open('./data.txt','r+') as f:
    datalist=[line.split('\n')[0].split(',') for line in f.readlines()]
    data=[list(map(float,li)) for li in datalist]


data=torch.tensor(data)
data_x=data[:,:-1].float()
data_y=data[:,-1].unsqueeze(1)

#数据预处理 缩放到0-1之间
x_min=torch.min(data_x,dim=0)[0]
x_max=torch.max(data_x,dim=0)[0]
data_x=(data_x-x_min)/(x_max-x_min)

net=nn.Linear(2,1,bias=True)
optimizer=optim.SGD(net.parameters(),lr=0.01)
criterion=nn.BCEWithLogitsLoss() #将sigmoid与loss结合起来

for e in range(10000):
    y_pred=net(data_x)
    loss=criterion(y_pred,data_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('loss:',loss.item())

w=net.weight.data
b=net.bias.data
x=np.arange(0,1,0.01)
y=(-float(w[:,0])*x-float(b[0]))/float(w[:,1])

data_x0=data_x[data_y[:,-1]==0,:]
data_x1=data_x[data_y[:,-1]==1,:]

plt.plot(data_x0[:,0].detach().numpy(),data_x0[:,1].detach().numpy(),'ro',label='x0')
plt.plot(data_x1[:,0].detach().numpy(),data_x1[:,1].detach().numpy(),'go',label='x1')
plt.plot(x,y,'b')
plt.legend()
plt.show()