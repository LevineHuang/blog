除了主流的 ResNet 流派和 Inception 流派不断追求更高的准确率，移动端的应用也是一大方向，比如 SqueezeNet、MobileNet v1 和 v2、ShuffleNet 等。

## **MobileNet v1：**2017

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

和 Xception 类似，通过 depthwise separable convolution 来减少计算量，设计了一个适用于移动端的，取得性能和效率间很好平衡的一个网络。

## **MobileNet v2：**2018

Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

使用了 ReLU6（即对 ReLU 输出的结果进行 Clip，使得输出的最大值为 6）适配移动设备更好量化，然后提出了一种新的 Inverted Residuals and Linear Bottleneck，即 ResNet 基本结构中间使用了 depthwise 卷积，一个通道一个卷积核，减少计算量，中间的通道数比两头还多（ResNet 像漏斗，MobileNet v2 像柳叶），并且全去掉了最后输出的 ReLU。具体的基本结构如下图右侧所示：

![从LeNet到SENet——卷积神经网络回顾](https://static.leiphone.com/uploads/new/article/740_740/201802/5a7811e6966f2.png?imageMogr2/format/jpg/quality/90)

## **ShuffleNet：**2017

ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

Xception 已经做得很好了，但是 1x1 那里太耗时间了成了计算的瓶颈，那就分组啦较少计算量，但是分组了，组和组之间信息隔离了，那就重排 shuffle 一下，强行让信息流动。具体的网络结构如上图左侧所示。channel shuffle 就是对通道进行重排，将每组卷积的输出分配到下一次卷积的不同的组去：

![从LeNet到SENet——卷积神经网络回顾](https://static.leiphone.com/uploads/new/article/740_740/201802/5a78120456220.png?imageMogr2/format/jpg/quality/90)

上图的 a 是没有 shuffle，效果很差，b 和 c 则是等价的有 shuffle 的。ShuffleNet 可以达到和 AlexNet 相同的精度，并且实际速度快 13 倍（理论上快 18 倍）。

## **SENet**

除了上面介绍的久经考验的网络以外，还有各种各样的新的网络，比如 NASNet、SENet、MSDNet 等等。其中，SENet 的 Squeeze-Excitation 模块在普通的卷积（单层卷积或复合卷积）由输入 X 得到输出 U 以后，对 U 的每个通道进行全局平均池化得到通道描述子（Squeeze），再利用两层 FC 得到每个通道的权重值，对 U 按通道进行重新加权得到最终输出（Excitation），这个过程称之为 feature recalibration，通过引入 attention 重新加权，可以得到抑制无效特征，提升有效特征的权重，并很容易地和现有网络结合，提升现有网络性能，而计算量不会增加太多。

![从LeNet到SENet——卷积神经网络回顾](https://static.leiphone.com/uploads/new/article/740_740/201802/5a78121b4296c.png?imageMogr2/format/jpg/quality/90)

SE module 是一个很通用的模块，可以很好地和现有网络集成，提升现有效果。