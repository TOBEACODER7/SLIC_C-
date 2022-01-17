# SLIC_C++


《数字图像处理课堂作业》

SLIC超像素分割算法复现

  学 院：	计算机学院   
  专 业：	计算机科学与技术
      	        
      	        

2022 年 1 月 14 日

1. 实验目的

阅读并理解发表在SLIC Superpixels Compared to State-of-the-Art Superpixel
Methods中的SLIC超像素分割算法，并使用C++语言进行代码的复现

1. 实验内容

理解实验原理、实验步骤，并进行代码的复现得出实验结果，结合实验结果进行分析。

要求源码+注释

1. 实验环境
   opencv 4.5.4版本
   C++作为编程语言
   使用VS2019作为IDE
2. 实验过程

PART 1 前期准备以及环境搭建

安装VS2019并配置C++开发环境

安装opencv 4.5.4版本并且在win10系统变量中添加文件路径

在VS2019中创建项目并且在项目属性中添加opencv库的链接路径

写一个简单的使用opencv库函数的demo看测试代码是否能够正常运行，若一切正常则环境搭建完毕

PART 2 SLIC超像素分割算法分析

超像素是2003年Xiaofeng
Ren提出和发展起来的图像分割技术，是指具有相似纹理、颜色、亮度等特征的相邻像素构成的有一定视觉意义的不规则像素块。

SLIC是一个超像素点分割算法，它是一种基于K-means的聚类算法，根据像素的颜色和距离特征进行聚类来实现良好的分割效果。

具体实现步骤：

1. 图像预处理
   将待处理图像又BGR颜色空间转换到Lab颜色空间。Lab颜色空间更符合人类对颜色的视觉感知，这样得出的聚类结果更加准确。
   Lab颜色空间具有三个通道，l表示亮度，数值范围[0,100];a表示从绿色到红色的分量，数值范围[-128,127];b表示从蓝色到黄色的分量，数值范围[-128,127];
   1. 初始化聚类中心
      根据参数确定需要划分多少个簇，假设共有N个像素点、K个簇，则每个簇的大小为N/K，相邻中心距离为S=Sqrt(N/K)
   2. 优化聚类中心
      为了防止聚类中心落在梯度较大的像素点处，即为了防止落在图像边界轮廓处，我们需要在聚类中心n*n领域内（一般情况下n=3）选择梯度值最小的像素点作为聚类中心。
      将图像看为二维离散函数，计算梯度也就是对这个函数的求导，则计算梯度的公式如下：
      
   3. 计算像素点与聚类中心的距离
      在聚类中心2S*2S的邻域内计算像素点与每个聚类中心的距离
      
      距离使用欧氏距离，总距离D由颜色距离dc和空间距离ds组成
      
      为了平衡颜色距离和空间距离的权重，在计算总距离时需要对距离空间进行归一化，给颜色空间也会除以一个m值来调节两者之间的影响权重，m取值范围[1,40]
      
      m取值对于超像素分割的影响：m越大，空间距离的权重越大，生成的像素会更趋于规则形状；m越小，颜色距离的权重越大，生成像素在边缘会更为紧凑，形状大小较为不规则。
   4. 像素点划分
      将每个像素点划分给距离其最近的聚类中心的簇。
   5. 更新聚类中心
      计算属于同一聚类的所有像素点的平均向量值，重新得到聚类中心。
   6. 循环迭代聚类过程4-6
      迭代终止的条件是簇的划分不再变化或者达到最大迭代次数，在这里最大迭代次数一般取10
2. 实验结果：
   实现了对于图像的超像素划分聚类，这里使用图像处理领域的标准图Lena来进行实验结果的展示



如上图所示，由于采用了计算梯度来优化聚类中心的方式，聚类中心的分布基本都未处于图像边界处。

关于实验结果的一些思考：为什么会出现远超于聚类中心数的块数划分？

由于我们计算五维向量间距离时采用了一个参数m来调整距离空间和颜色空间的权重，这里我选择了m=10作为权重调整，这个值的选取较为随意。在实际的距离划分中，会出现某个像素距离聚类中心的总距离小于另一个像素，但是其空间距离却大于这个像素，所以出现了看似远超于聚类中心的块数。


