# 算法方案介绍

## 技术背景

通过摄像头识别出自定义的 5 个以上手势，手势距离摄像头 1.5 米。


## 方案原理

以裁剪版的MobileNet为特征提取器以减少计算量，修改端到端的目标检测方法YOLO v2，完成手势检测任务。  
MobileNet原理详见[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  
YOLO v2原理详见[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)


## 实施细节

本项目主要基于[PyTorch](http://pytorch.org/)进行模型搭建，训练以及测试。完成训练后模型转换成Caffe下的prototxt与caffemodel格式，然后在嵌入式平台上的CaffeOnACL架构下运行。  
Pytorch下MobileNet代码实现主要基于[pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet)修改而来。
裁剪后的MobileNet网络结构如下表所示：  
Type/Stride | Filter Shape  
--- | ---   
Conv/s1 | 3x3x3x10  
Conv dw/s2 | 3x3x10 dw  
Conv/s1 | 1x1x10x10  
Conv dw/s1 | 3x3x10 dw  
Conv/s1 | 1x1x10x10  
Conv dw/s2 | 3x3x10 dw  
Conv/s1 | 1x1x10x20  
Conv dw/s1 | 3x3x20 dw  
Conv/s1 | 1x1x20x20  
Conv dw/s2 | 3x3x20 dw  
Conv/s1 | 1x1x20x40   
Conv dw/s1 | 3x3x40 dw  
Conv/s1 | 1x1x40x40  
Conv dw/s2 | 3x3x40 dw  
Conv/s1 | 1x1x40x80 dw  
Conv dw/s1 | 3x3x80 dw  
Conv/s1 | 1x1x80x80 dw  
Conv dw/s1 | 3x3x80 dw  
Conv/s1 | 1x1x80x80 dw  
Conv dw/s1 | 3x3x80 dw  
Conv/s1 | 1x1x80x80 dw  
Conv dw/s1 | 3x3x80 dw  
Conv/s1 | 1x1x80x80 dw  
Pytorch下YOLO v2 模型搭建以及训练部分代码实现基于[yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)修改而来。适应技术背景，使用一个anchor。结合测试情况，将人脸也列入分类类别中，于是一共有6类（5个静态手势类以及1个脸部类）。于是最终需要4个bounding box位置回归量，1个objectness量， 以及6个类别i评分量，共11个量。修改特征提取器（MobileNet）后YOLO v2的分类回归层参数为：  
Type/Stride | Filter Shape  
--- | ---   
Conv/s1 | 3x3x80x80  
Conv/s1 | 3x3x80x80  
Conv/s1 | 1x1x80x11  
训练好原始模型后，基于模型中Conv层与BN层相邻的特性，将这两层参数结合从而减少inference时的计算量。  
模型在PyTorch上训练好后，为了能够在RK3399上CaffeOnACL下运行，需要转换成Caffe的模型格式，这里利用[pytorch2caffe](https://github.com/longcw/pytorch2caffe)进行格式转换。  
demo程序中，网络的初始输入维度为240x320x3(480x640x3降采样得到)，检测到只有一个手势后，收敛检测区域为之前手势周围的144x192的范围，从而减少网络输入的初始维度，显著降低降低计算量。


## 数据集

训练（train）数据集使用真实数据，对3个人在室内，3个人在室外进行旋转360度的拍摄。对每个环境下每人每个手势截取300帧。基于[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)，利用[PyOpenPose](https://github.com/FORTH-ModelBasedTracker/PyOpenPose)自动对图片进行手势以及脸部进行标注。  
验证（val）数据集也使用真实数据，对1个人在室内进行固定背景下的拍摄，对每个手势取200帧并进行自动标注。  
测试（test）数据集来自对1个人在决赛背景下的5种手势的视频，每个手势100帧。  
上述数据集都由手机后置镜头拍摄而来，比例为16：9，处理后降采样为426:240。在训练和验证时，利用自己编写的dataloader从相应图像中随机截取包含手势的192x144（4：3）大小的图像。测试数据集则遵循COCO数据集格式以方便调用COCO的Python [API](https://github.com/cocodataset/cocoapi)进行AP（Average Precision）以及AR（Average Recall）的测试。


## 算法训练

基于[DSOD: Learning Deeply Supervised Object Detectors from Scratch](https://arxiv.org/abs/1708.01241)的思想直接利用Detection数据集对网络从头进行训练。  
采用Adadelta优化，观察训练损失以及验证损失，人为选择学习率退火规则。
在训练集上完成训练后，在具有一定手势旋转的验证集上对模型最后三层进行finetuning以提升网络对手势旋转的鲁棒性。



# 测试数据及验证

## 评估方法

基于COCO数据集格式的决赛背景下的手势测试数据集，用训练好的模型进行测试，并按照COCO result文件的格式保存Detection的结果，利用[cocoapi](https://github.com/cocodataset/cocoapi)对Detection结果进行评估。

## 测试数据分析

选择不同的可信度阈值，得到测试结果为：
* threshold=0.75:  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.246  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.387  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.268  
**Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.322**  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.203  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.272  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.272  
**Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350**  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.217  
* threshold=0.5:  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.291  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.498  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.295  
**Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370**   
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.262  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.336  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.336  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336  
**Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.428**  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.287  
* threshold=0.25:    
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.539  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.298  
**Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380**  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.281  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.360  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360  
**Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.456**  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.314  
* threshold=0.1:      
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.307  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.550  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.298  
**Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.385**  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.284  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.367  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367  
**Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.479**   
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320  
由此可见，模型可信度阈值在0.1的时候模型测试效果最好，AP与AR都为最高。




# 总结

## 算法特点

* 深度学习
* 端到端

## 与demo算法横向比较-优缺点
* 优点:  
  训练速度快
* 缺点:  
    计算量较大，速度较慢
