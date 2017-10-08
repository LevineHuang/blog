
window选择:
+ 固定时间跨度
+ 固定数据长度，数据个数
+ window内的数据个数 + 数据方差


### LSTM的应用
#### 序列分类
+ http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
预测随空间或时间变化的序列的类别

电影评论情感分类

word embedding

One-hot Representation
把每个词表示为一个很长的向量。这个向量的维度是词表大小，其中绝大多数元素为 0，只有一个维度的值为 1，这个维度就代表了当前的词。

缺点：
维数灾难[Bengio 2003]
“词汇鸿沟”现象：任意两个词之间都是孤立的。光从这两个向量中看不出两个词是否有关系，哪怕是话筒和麦克这样的同义词也不能幸免于难。

Distributed representation
 最大的贡献就是让相关或者相似的词，在距离上更接近了。向量的距离可以用最传统的欧氏距离来衡量，也可以用 cos 夹角来衡量。
