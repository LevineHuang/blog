
###  Language Model
to predict the probability of observing the sentence (in a given dataset) as:

\begin{aligned}  P(w_1,...,w_m) = \prod_{i=1}^{m} P(w_i \mid w_1,..., w_{i-1})  \end{aligned}

计算一个句子出现概率有什么作用s
+ 翻译
+ 语音识别
+ generative model - Because we can predict the probability of a word given the preceding words, we are able to generate new text

缺点：当前置单词很多时，RNN遇到困难，引出LSTM。

### 训练数据及其预处理
15000条评论数据

#### 分词  tokenize text
借助NLTK中的word_tokenize和sent_tokenize方法

#### 去除低频词
原因：
+ 太多的词，模型难以训练
+ 低频词没有足够的上下文样本数据

低频词的处理方式：
按词频选取前N个（如8000），其它即为低频词，用“UNKNOWN_TOKEN”代替。生产新文本后，随机取词汇库外的一个词替换“UNKNOWN_TOKEN”，或者不断生产新文本，直到生成文本中不包含“UNKNOWN_TOKEN”为止。


### 梯度消失问题 THE VANISHING GRADIENT PROBLEM
+ W矩阵初始化
+ 正则化 用ReLU代替tanh或 sigmoid激活函数】
+ 采用LSTM或GRU(Gated Recurrent Unit, LSTM的简化版本)结构


## Part4
### LSTM网络
通过gating mechanism机制解决RNNs的梯度消失问题。
plain RNNs可以看做是LSTMs的特殊形式(input gate 取1，forget gate 取0，output gate 取1)
