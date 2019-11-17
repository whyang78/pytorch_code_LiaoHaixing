import torch
from torch import nn,optim
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(78)
lr=0.001

#************************使用梯度下降原理更新参数***************************#

#一维线性回归
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)

def linear_model(x,w,b):
    return x.mm(w)+b

def loss_func(y,y_):
    return  torch.mean((y-y_)**2)

w=torch.randn(1,1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)

for e in range(1000):
    y_fit=linear_model(x_train,w,b)
    loss=loss_func(y_train,y_fit)

    loss.backward()

    w.data=w.data-lr*w.grad.data
    b.data=b.data-lr*b.grad.data

    #刚开始这两个参数都没有梯度，所以每次都在最后将其梯度清0
    w.grad.zero_()
    b.grad.zero_()
    print('loss:',loss.item())

plt.scatter(x_train.detach().numpy(),y_train.detach().numpy(),c='g')
plt.plot(x_train.detach().numpy(),linear_model(x_train,w,b).detach().numpy(),color='r',label='fitting curve')
plt.legend()
plt.show()

#多项式回归
# 定义一个多变量函数
w_target = np.array([0.5, 3, 2.4], dtype=np.float32) # 定义参数
b_target = np.array([0.9], dtype=np.float32) # 定义参数
f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2]) # 打印出函数的式子
print(f_des)

x_sample = np.arange(-3, 3, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
x_sample=np.stack([x_sample**i for i in range(1,4)],axis=1)

x_sample=torch.from_numpy(x_sample).float()
y_sample=torch.from_numpy(y_sample).float().unsqueeze(1)

w=torch.randn(3,1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def mutil_model(x,w,b):
    return torch.mm(x,w)+b

def loss_func(y,y_):
    return  torch.mean((y-y_)**2)

for e in range(1000):
    y_fit=mutil_model(x_sample,w,b)
    loss=loss_func(y_sample,y_fit)

    loss.backward()
    w.data=w.data-lr*w.grad.data
    b.data=b.data-lr*b.grad.data

    w.grad.zero_()
    b.grad.zero_()

    print('loss:',loss.item())

plt.plot(x_sample.detach().numpy()[:,0], y_sample.detach().numpy(), color='g',label='real curve')
plt.plot(x_sample.detach().numpy()[:,0],mutil_model(x_sample,w,b).detach().numpy(),color='r',label='fitting curve')
plt.legend()
plt.show()


#*****************************使用全连接网络**************************#
#多项式回归拟合
net=nn.Linear(3,1,bias=True)
criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=lr)

for e in range(1000):
    pred=net(x_sample)
    loss=criterion(pred,y_sample)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('loss:',loss)

print('w:{},b:{}'.format(net.weight.data,net.bias.data))


