[TOC]

根据泛逼近定理（universal approximation theorem），如果给定足够的容量，一个单层的前馈网络就足以表示任何函数。但是，这个层可能是非常大的，而且网络容易过拟合数据。因此，研究界有一个共同的趋势，就是网络架构需要更深。

从 AlexNet 的提出以来，state-of-the art 的 CNN 架构都是越来越深。虽然 AlexNet 只有5层卷积层，但后来的 VGG 网络[3]和 GoogLeNet（也作 Inception_v1）[4]分别有19层和22层。

但是，如果只是简单地将层堆叠在一起，增加网络的深度并不会起太大作用。这是由于难搞的梯度消失（vanishing gradient）问题，深层的网络很难训练。因为梯度反向传播到前一层，重复相乘可能使梯度无穷小。结果就是，随着网络的层数更深，其性能趋于饱和，甚至开始迅速下降。

## **ResNet：**2016

Deep residual learning for image recognition

ImageNet Top5 错误率 3.57%

2015年何恺明推出的ResNet在ISLVRC和COCO上横扫所有选手，获得冠军。ResNet在网络结构上做了大创新，而不再是简单的堆积层数，ResNet在卷积神经网络的新思路，绝对是深度学习发展历程上里程碑式的事件。

![img](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180217131926202-233779647.jpg)

闪光点：

- 层数非常深，已经超过百层
- 引入残差单元来解决退化问题

从前面可以看到，随着网络深度增加，网络的准确度应该同步增加，当然要注意过拟合问题。但是网络深度增加的一个问题在于这些增加的层是参数更新的信号，因为梯度是从后向前传播的，增加网络深度后，比较靠前的层梯度会很小。这意味着这些层基本上学习停滞了，这就是梯度消失问题。

深度网络的第二个问题在于训练，当网络更深时意味着参数空间更大，优化问题变得更难，因此简单地去增加网络深度反而出现更高的训练误差，深层网络虽然收敛了，但网络却开始退化了，即增加网络层数却导致更大的误差，比如下图，一个56层的网络的性能却不如20层的性能好，这不是因为过拟合（训练集训练误差依然很高），这就是烦人的退化问题。残差网络ResNet设计一种残差模块让我们可以训练更深的网络。

![img](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180217131941296-1327847371.png)

这里详细分析一下残差单元来理解ResNet的精髓。

从下图可以看出，数据经过了两条路线，一条是常规路线，另一条则是捷径（shortcut），直接实现单位映射的直接连接的路线，这有点类似与电路中的“短路”。通过实验，这种带有shortcut的结构确实可以很好地应对退化问题。我们把网络中的一个模块的输入和输出关系看作是y=H(x)，那么直接通过梯度方法求H(x)就会遇到上面提到的退化问题，如果使用了这种带shortcut的结构，那么可变参数部分的优化目标就不再是H(x),若用F(x)来代表需要优化的部分的话，则H(x)=F(x)+x，也就是F(x)=H(x)-x。因为在单位映射的假设中y=x就相当于观测值，所以F(x)就对应着残差，因而叫残差网络。为啥要这样做，因为作者认为学习残差F(X)比直接学习H(X)简单！设想下，现在根据我们只需要去学习输入和输出的差值就可以了，绝对量变为相对量（H（x）-x 就是输出相对于输入变化了多少），优化起来简单很多。

考虑到x的维度与F(X)维度可能不匹配情况，需进行维度匹配。这里论文中采用两种方法解决这一问题(其实是三种，但通过实验发现第三种方法会使performance急剧下降，故不采用):

- zero_padding:对恒等层进行0填充的方式将维度补充完整。这种方法不会增加额外的参数
- projection:在恒等层采用1x1的卷积核来增加维度。这种方法会增加额外的参数

![img](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180217131952952-92773471.png)

下图展示了两种形态的残差模块，左图是常规残差模块，有两个3×3卷积核卷积核组成，但是随着网络进一步加深，这种残差结构在实践中并不是十分有效。针对这问题，右图的“瓶颈残差模块”（bottleneck residual block）可以有更好的效果，它依次由1×1、3×3、1×1这三个卷积层堆积而成，这里的1×1的卷积能够起降维或升维的作用，从而令3×3的卷积可以在相对较低维度的输入上进行，以达到提高计算效率的目的。

![img](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180217132002999-1852938927.png)



## **DenseNet：**2016

Densely Connected Convolutional Networks

DenseNet 将 residual connection 发挥到极致，每一层输出都直连到后面的所有层，可以更好地复用特征，每一层都比较浅，融合了来自前面所有层的所有特征，并且很容易训练。**不同于 ResNet 将输出与输入相加，形成一个残差结构，DenseNet 将输出与输入相并联，实现每一层都能直接得到之前所有层的输出。**

DenseNet 进一步利用 shortcut 连接的好处——将所有层都直接连接在一起。在这个新架构中，每层的输入由所有前面的层的特征映射（feature maps）组成，其输出传递给每个后续的层。特征映射与 depth-concatenation 聚合。

![从LeNet到SENet——卷积神经网络回顾](https://static.leiphone.com/uploads/new/article/740_740/201802/5a78110cc1142.png?imageMogr2/format/jpg/quality/90)

ResNet 将输出与输入相加，形成一个残差结构；而 DenseNet 却是将输出与输入相并联，实现每一层都能直接得到之前所有层的输出。

Aggregated Residual Transformations for Deep Neural Networks [8]的作者除了应对梯度消失问题外，还认为这种架构可以鼓励特征重新利用，从而使得网络具有高度的参数效率。一个简单的解释是，在 Deep Residual Learning for Image Recognition [2]和 Identity Mappings in Deep Residual Networks [7]中，Identity Mapping 的输出被添加到下一个块，如果两个层的特征映射具有非常不同的分布，这可能会阻碍信息流。因此，级联特征映射可以保留所有特征映射并增加输出的方差，从而鼓励特征重新利用。

![img](http://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2gvSQN9s4kYQUPQjspFl4ibOWLbA5yaZk44GsKBy0FS9SfIqo07Krv16TyCFLrLTVIoyQwvX3nllQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



遵循这个范式，我们知道第 l 层将具有 k *（l-1）+ k_0 个输入特征映射，其中 k_0 是输入图像中的通道数。作者使用一个名为增长率（k）的超参数来防止网络的生长过宽，以及使用一个 1x1 的卷积瓶颈层来减少昂贵的 3x3 卷积之前的特征映射数量。整体结构如下表所示：

![img](http://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2gvSQN9s4kYQUPQjspFl4ib3Bq4N2xzG0xZvDlHnW2wRzpYjpSoc8DMqt71ugmibKHWnVQicyjzleZw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

DenseNet的架构

缺点是显存占用更大并且反向传播计算更复杂一点。

## WRN(Wide Residual Network): 2016

从“宽度”入手做提升

Wide Residual Network（WRN） 由 Sergey Zagoruyko 和 Nikos Komodakis 提出。虽然网络不断向更深层发展，但有时候为了少量的精度增加需要将网络层数翻倍，这样减少了特征的重用，也降低训练速度。因此，**作者从“宽度”的角度入手，提出了 WRN，16 层的 WRN 性能就比之前的 ResNet 效果要好**。

![img](http://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb255bjBF6Chqe75O08lIib7IrQnBlrp52BfB0aBohmlR7HZIB1gH7Ia1EN1MeZCmm1Liad442G9Wjiag/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

上图中 a，b 是何恺明等人提出的两种方法，b 计算更节省，但是 WDN 的作者想看宽度的影响，所以采用了 a。作者提出增加 residual block 的 3 种简单途径： 

1. 更多卷积层 

2. 加宽（more feature planes） 

3. 增加卷积层的滤波器大小（filter sizes）

![img](http://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb255bjBF6Chqe75O08lIib7I2t9ht5LaFpfxM8Qicg3rpXfZCNibJ2IPUMfMBQcu4TzSOdEStvxJEaFA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

WRN 结构如上，作者表示，小的滤波器更加高效，所以不准备使用超过 3x3 的卷积核，提出了宽度放大倍数 k 和卷积层数 l。

作者发现，参数随着深度的增加呈线性增长，但随着宽度却是平方长大。虽然参数会增多，但卷积运算更适合 GPU。参数的增多需要使用正则化（regularization）减少过拟合，何恺明等人使用了 batch normalization，但由于这种方法需要heavy augmentation，于是作者使用了 dropout。

![img](http://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb255bjBF6Chqe75O08lIib7IsZCaThO45k65siclC4dOI2cKkQs1QeOo3ictUFD5WNjCrkZicgFXBpSvA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

WRN 40-4 与 ResNet-1001 结果相似，参数数量相似，但是前者训练快 8 倍。作者总结认为：

1. 宽度的增加提高了性能 

2. 增加深度和宽度都有好处，直到参数太大，正则化不够 

3. 相同参数时，宽度比深度好训练 

## **ResNeXt：**2017

Aggregated Residual Transformations for Deep Neural Networks

ImageNet Top5 错误率：3.03%

Inception 借鉴 ResNet 得到 Inception-ResNet，而 ResNet 借鉴 Inception 得到了 ResNeXt，对于每一个 ResNet 的每一个基本单元，横向扩展，将输入分为几组，使用相同的变换，进行卷积：

![从LeNet到SENet——卷积神经网络回顾](https://static.leiphone.com/uploads/new/article/740_740/201802/5a7811207f26c.png?imageMogr2/format/jpg/quality/90)

上面左边是 ResNet，右边是 ResNeXt，通过在通道上对输入进行拆分，进行分组卷积，每个卷积核不用扩展到所有通道，可以得到更多更轻量的卷积核，并且，卷积核之间减少了耦合，用相同的计算量，可以得到更高的精度。

这个可能看起来很眼熟，因为它与 GoogLeNet [4]的 Inception 模块非常类似。它们都遵循“拆分-转换-合并“的范式，区别只在于 ResNeXt 这个变体中，不同路径的输出通过将相加在一起来合并，而在 GoogLeNet [4]中不同路径的输出是深度连结的。另一个区别是，GoogLeNet [4]中，每个路径彼此不同（1x1, 3x3 和 5x5 卷积），而在 ResNeXt 架构中，所有路径共享相同的拓扑。

**ResNeXt 的作者引入了一个被称为“基数”（cardinality）的超参数——即独立路径的数量，以提供一种新方式来调整模型容量。实验表明，通过增加“基数”提高准确度相比让网络加深或扩大来提高准确度更有效。**作者表示，基数是衡量神经网络在深度（depth）和宽度（width）之外的另一个重要因素。作者还指出，与 Inception 相比，这种新的架构更容易适应新的数据集/任务，因为它有一个简单的范式，而且需要微调的超参数只有一个，而 Inception 有许多超参数（如每个路径的卷积层核的大小）需要微调。

这一新的构建块有如下三种对等形式：

![img](http://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2gvSQN9s4kYQUPQjspFl4ibZ8USvK5OgPEiaZ01gdjY2Hj3txwrxcaGBdDnGiaH3toVUaLUs28BBRtA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



在实践中，“分割-变换-合并”通常是通过逐点分组卷积层来完成的，它将输入分成一些特征映射组，并分别执行卷积，其输出被深度级联然后馈送到 1x1 卷积层。

在 ImageNet-1K 数据集上，作者表明，即使在保持复杂性的限制条件下，增加基数也能够提高分类精度。此外，当增加容量时，增加基数比更深或更宽更有效。ResNeXt 在 2016 年的 ImageNet 竞赛中获得了第二名。

## DPN: 2017

结合残差网络与 DenseNet 两者优点，夺得 ImageNet 2017 目标定位冠军

新加坡国立大学与奇虎 AI 研究院合作，指出 ResNet 是 DenseNet 的一种特例，并提出了一类新的网络拓补结构：双通道网络（Dual Path Network）。在 ImageNet-1k 分类任务中：该网络不仅提高了准确率，还将200 层 ResNet 的计算量降低了 57%，将最好的 ResNeXt (64x4d) 的计算量降低了25%；131 层的 DPN 成为新的最佳单模型，并在实测中提速约 300%。

**作者发现，Residual Networks 其实是 DenseNet 在跨层参数共享时候的特例。于是，他们结合残差网络和 DenseNet 两者的优点，提出了一类全新的双通道网络结构：Dual Path Network（DPNs）。**

![img](http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2eSOPSqHo2WFyGyGibPe4rhTH6FUE6PZLiaibhg8eth227ER0XGgv6WHSCcnR5zmCNSBeARoNdVP1ew/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

DPN 具体网络结构

其核心思想是，将残差通道和 densely connected path 相融合，实现优缺互补，其重点不在于细节部分是如何设定的。

作者分别在“图像分类”，“物体检测”和“物体分割”三大任务上对 DPN 进行了验证。在 ImageNet 1000 类分类任务中的性能如表 2 所示：

![img](http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2eSOPSqHo2WFyGyGibPe4rhl2bZ2iaiaOS8D2ZtOTJaKiaQJ7k4tyGXrE3dwuv7Via82Wp86zNcbRWJGw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在实测中： DPN-98 也显著提高了训练速度，降低内存占用，并保持更高的准确率。在 ImageNet-1k 分类任务中：该网络不仅提高了准确率，还将200 层 ResNet 的计算量降低了 57%，将最好的 ResNeXt (64x4d) 的计算量降低了25%；131 层的 DPN 成为新的最佳单模型，并在实测中提速约 300%。

就在这周，ImageNet 官方网站公布了 2017 年ImageNet Large Scale Visual Recognition Challenge 2017 (ILSVRC2017) 的比赛结果，在目标定位任务中，新加坡国立大学与奇虎360 合作提出的 DPN 双通道网络 + 基本聚合获得第一，定位错误率为 0.062263。

## 参考

1. [CNN网络架构演进：从LeNet到DenseNet](https://www.cnblogs.com/skyfsm/p/8451834.html)
2. [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
3. [ResNet 6大变体](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652001197&idx=1&sn=4239318655806de8ed807d44cdb1b99c&chksm=f121275cc656ae4a3ad2dedc3b7a53b57fe92f76b97fc1c237bcf9e6a4cdb8adba67ff470df6&mpshare=1&scene=1&srcid=07225oxQWWm09aX1Gk8dM717#rd)