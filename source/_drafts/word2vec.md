## 词向量

one hot编码，输入DNN网络，从隐藏层到输出的softmax层的计算量很大，因为要计算所有词的softmax概率，再去找概率最大的值。

- CBOW(Continuous Bag-of-Words）

  模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。

- Skip-Gram

  模型的训练输入是特定的一个词的词向量，输出是softmax概率排前8的8个词，对应的Skip-Gram神经网络模型输入层有1个神经元，输出层有词汇表大小个神经元。V是词汇表的大小，输入为每个词的onehot 编码。

![](imgs/word2vec-DNN.png)