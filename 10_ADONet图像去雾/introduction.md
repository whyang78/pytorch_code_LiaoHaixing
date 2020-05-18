### `文件说明`  
1. 本实验是使用ADONet对图片进行去雾处理。不涉及训练，直接读取预训练权重 `dehazer.pth`。  
2. `test_images`文件夹是测试原数据， `result`文件夹是去雾结果。  
3. `clean.py`是主程序。  
    + AODNet网络搭建  
    + Dateset创建类，返回数据及对应路径  
4. 参考网址：  
    https://blog.csdn.net/qq_35608277/article/details/86010157