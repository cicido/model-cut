# model-cut
tensorflow下实现何宜辉的channel-pruning, 只涉及到卷积层，池化，relu层。
原作者git: https://github.com/yihui-he/channel-pruning

由于是优化风格迁移，因而模型来自于：https://github.com/jonrei/tf-AdaIN/blob/master/AdaIN.py

主要思路（参照何宜辉代码）：
1. 生成输入数据
2. 采样，生成X,Y. 
3. 抽取模型参数,得到W
4. 每两个卷积层做一次裁剪.

原作者中VH_decompose, channel-decompase并没有使用。只用到chanel pruning. 
主要工作在构造原作者代码中的W,X,Y这三个Tensor
