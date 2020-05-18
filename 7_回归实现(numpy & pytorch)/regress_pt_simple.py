import torch
from matplotlib import pyplot as plt


x=torch.linspace(-1,1,100).reshape(100,1)
y=3 * torch.pow(x,2) + 2 + 0.2 * torch.rand(x.size())

w=torch.randn(1,1,requires_grad=True)
b=torch.zeros(1,1,requires_grad=True)

lr=0.001
for e in range(1000):
    y_pred=torch.pow(x,2).mm(w) + b
    loss=0.5 * torch.sum(torch.pow(y_pred-y,2))

    loss.backward()
    with torch.no_grad():
        w-=lr * w.grad
        b-=lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

plt.figure()
plt.plot(x.numpy(),y_pred.detach().numpy(),'r-',label='predict')
plt.scatter(x.numpy(),y.numpy(),c='b',marker='o',label='true')
plt.legend()
plt.show()

print([w,b])

