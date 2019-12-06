1. 自动编码器重建mnist数据集，详见 `autoencoder.py`  
    + 分别搭建全连接层和卷积层实现的自编码器，并保存每个epoch后的生成图片于 `sample_autoencoder`  
    + 编码设置为3维，并可视化三维分布  
2. 变分自动编码器重建mnist数据集，详见 `varoational_autoencoder.py`  
    + 搭建全连接层实现的变分自动编码器  
    + 保存图片  
3. GAN学习mnist，详见 `gan.py`  
    + 使用全连接层编写生成器与判别器  
    + 使用visdom显示loss曲线与生成图像  
4. DCGAN学习mnist，详见 `dcgan.py`  
    + 使用卷积层与反卷积编写生成器与判别器  
    + 使用visdom显示loss曲线与生成图像  
5. WGAN学习mnist，详见 `wgan.py`  
    + 使用卷积层编写生成器与判别器  
    + 损失函数中添加gradient_penalty项  
    + 使用visdom显示loss曲线与生成图像  
    + 个人认为不太严谨  
6. WGAN学习，代码较为准确。  
    + WGAN配合权重裁剪，详见 `wgan_clipping.py`    
    + WGAN配合gradient_penalty，详见 `wgan_gradient_penalty.py`  
   
