#实战之猫狗大战  
    采用了三种方法进行训练测试。  
1. 迁移学习，详见 `fix_train.py`  
    + 使用预训练resnet18网络进行迁移学习  
    + 采用两种方法：  
        * 冻结除最后一层全连接层的其他网络层，只训练全连接层。  
        * 所有层都训练。此时优化器的学习率对全连接层和其它层分别设置。 
2. 多模型训练测试，详见 `feature_train.py`  
    + 使用三种预训练网络分别对训练集和测试集进行特征提取，切记不要shuffle，详见 `feature_extraction.py`,  
        提取特征存放在`feature`文件夹中。    
    + 制作数据集，将同一对象的特征进行拼接，详见 `dataset.py`  
    + 特征提取网络与分类网络定义详见 `net.py`  
---  
###多模型训练测试过程  
1. 首先对原本的kaggle cat vs dog 数据集进行处理，将数据集中的train按照类别  
    划分成训练集与测试集,详见 `preprocess.py`  
2. 特征提取  
    默认batchsize=5,命令行输入运行`feature_extraction.py`。以使用vgg16模型提取特征为例。  
    ```
    python --model vgg16 --phase train  
    python --model vgg16 --phase val 
    ```  
3. 运行`feature_train.py`