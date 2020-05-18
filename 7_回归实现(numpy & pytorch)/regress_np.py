import numpy as np
from matplotlib import pyplot as plt

x=np.linspace(-1,1,100).reshape(100,1)
y=3 * np.power(x,2) + 2 + 0.2 * np.random.rand(x.size).reshape(100,1)

w=np.random.rand(1,1)
b=np.random.rand(1,1)

lr=0.001
for e in range(1000):
    y_pred=np.power(x,2) * w + b
    loss=0.5 * np.sum(np.power(y_pred-y,2))

    grad_w=np.sum((y_pred-y) * np.power(x,2))
    grad_b=np.sum((y_pred-y))

    w-=lr * grad_w
    b-=lr * grad_b

plt.figure()
plt.plot(x,y_pred,'r-',label='predict')
plt.scatter(x,y,c='b',marker='o',label='true')
plt.legend()
plt.show()

print([w,b])