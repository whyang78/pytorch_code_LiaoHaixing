import torch
from torch import nn,optim
from torch.utils.data import DataLoader,TensorDataset
from torch.nn import functional as F
from collections import OrderedDict
from matplotlib import pyplot as plt

epoch=1000
bs=10
lr=0.01

x=torch.linspace(-1,1,100).reshape(100,1)
y=3 * torch.pow(x,2) + 2 + 0.2 * torch.rand(x.size())
dataset=TensorDataset(x,y)
dataloader=DataLoader(dataset,batch_size=bs,shuffle=True)

def adjust_lr(optimizer,gamma=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr']*=gamma

# 此网络没法得到具体的 w , b
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            OrderedDict([
                ('hidden',nn.Linear(1,20)),
                ('activate',nn.ReLU()),
                ('output',nn.Linear(20,1))
            ])
        )

    def forward(self, x):
        return self.net(x)


# 此网络可以得到具体的 w , b
w=nn.Parameter(torch.randn(1,1))
b=nn.Parameter(torch.zeros(1,1))
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # 网络中注册参数
        self.register_parameter('w',w)
        self.register_parameter('b',b)

    def forward(self, x):
        pow_x=torch.pow(x,2)
        output=F.linear(pow_x,weight=w,bias=b)
        return output

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=Net1().to(device)
criterion=nn.MSELoss().to(device)
optimizer=optim.SGD(model.parameters(),lr=lr)

for e in range(epoch):
    model.train()
    if (e+1)%500==0:
        adjust_lr(optimizer)
    for i,(batch_x,batch_y) in enumerate(dataloader):
        batch_x,batch_y=batch_x.to(device),batch_y.to(device)
        output=model(batch_x)
        loss=criterion(output,batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%5==0:
            print('epoch:{},index:{},loss:{:.4f},w:{:.4f},b:{:.4f}'.format(e,i,loss.item(),w.item(),b.item()))

plt.figure()
y_pred= torch.pow(x,2).mm(w.detach().cpu()) + b.detach().cpu()
plt.plot(x.numpy(),y_pred.numpy(),'r-')
plt.scatter(x.numpy(),y.numpy(),c='b',marker='o')
plt.show()
