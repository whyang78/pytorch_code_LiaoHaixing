##注意事项  
1. batchnorm批标准化的实现与应用，详见 `batch-normalization.ipynb`  
2. 学习网络结构时，可以参照torchvision.models中相应模型实现的源代码。  
3. Lenet网络实现，详见 `LeNet.py`  
    + 打印网络结构  
    + 利用torchsummary打印输入输出数据维度信息  
4. AlexNet网络实现，详见 `AlexNet.py`  
    + 打印网络结构  
    + 利用torchsummary打印输入输出数据维度信息  
    + 相比论文中，在全连接层之前添加了ROI pooling层，使得全连接层的输入维度是固定的
5. VGGNet网络实现，详见 `VGGNet.py`  
    + 实现VGG16，打印网络结构  
    + 利用torchsummary打印输入输出数据维度信息  
    + 相比论文中，在全连接层之前添加了ROI pooling层，使得全连接层的输入维度是固定的
    + 网络参数初始化  
6. ResNet网络实现，详见 `ResNet.py`  
    + 实现ResNet18，打印网络结构  
    + 利用torchsummary打印输入输出数据维度信息  
7. 使用cifar10数据集测试自定义resnet18网络，详见 `cifar10.py`  
    + 学习率调整  
    + 没有改变cifar10的输入尺寸，原网络输入224*224，因为实现时在全连接层加入了自适应pool层，  
      所以没有收到输入尺寸的影响。  
    + 建议训练前进行参数初始化，否则收敛太慢