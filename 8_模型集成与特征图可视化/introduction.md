### `文档说明`  
1. `visual_featuremap.py`对特征图可视化，以cifar10数据集为例。  
    特征图可视化其实是对每一个卷积核得到的特征（卷积层）进行可视化。  
2. `ensemble_model.py`集成多个模型对cifar10数据集进行预测。  
    + 模型详见 `mynet.py`  
    + 投票机制采用带权和不带权两种方式。