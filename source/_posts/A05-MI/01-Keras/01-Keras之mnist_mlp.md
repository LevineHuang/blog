---
title: 01-Keras之用MNIST数据集训练一个DNN
date: 2017-01-19 16:30:00
updated	: 2017-01-19 16:30:00
permalink: abc
tags:
- Keras
- Deep Learning
- AI
- MI
categories:
- DeepLearning
- Keras
---

## 01-Keras之用MNIST数据集训练一个DNN
----

#### 模型code
```python
# -*- coding: utf-8 -*-

'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 训练数据 60000张手写图片，28*28*1
# 测试数据 10000张手写图片，28*28*1

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化到0-1
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# to_categorical(y, nb_classes=None)
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
# y: 类别向量; nb_classes:总共类别数
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Dense层:即全连接层
# keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)


model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递activation参数实现。
# 以下两行等价于：model.add(Dense(512,activation='relu'))
model.add(Dense(512))
model.add(Activation('relu'))

# Dropout  需要断开的连接的比例
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# 打印出模型概况
print('model.summary:')
model.summary()

# 在训练模型之前，通过compile来对学习过程进行配置
# 编译模型以供训练
# 包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']
# 如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 训练模型
# Keras以Numpy数组作为输入数据和标签的数据类型
# fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
# nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。

# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))


# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# 按batch计算在某些输入数据上模型的误差
print('-------evaluate--------')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

```

#### 模型运行结果
```sh
Using TensorFlow backend.
60000 train samples
10000 test samples
model.summary:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_1 (Dense)                  (None, 512)           401920      dense_input_1[0][0]              
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 512)           0           activation_1[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 512)           262656      dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 512)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           activation_2[0][0]               
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            5130        dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10)            0           dense_3[0][0]                    
====================================================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 8s - loss: 0.2444 - acc: 0.9243 - val_loss: 0.1180 - val_acc: 0.9642
Epoch 2/20
60000/60000 [==============================] - 8s - loss: 0.1009 - acc: 0.9691 - val_loss: 0.0810 - val_acc: 0.9756
Epoch 3/20
60000/60000 [==============================] - 8s - loss: 0.0746 - acc: 0.9771 - val_loss: 0.0782 - val_acc: 0.9767
Epoch 4/20
60000/60000 [==============================] - 8s - loss: 0.0590 - acc: 0.9825 - val_loss: 0.0783 - val_acc: 0.9774
Epoch 5/20
60000/60000 [==============================] - 8s - loss: 0.0513 - acc: 0.9847 - val_loss: 0.0823 - val_acc: 0.9792
Epoch 6/20
60000/60000 [==============================] - 8s - loss: 0.0453 - acc: 0.9867 - val_loss: 0.0769 - val_acc: 0.9812
Epoch 7/20
60000/60000 [==============================] - 8s - loss: 0.0380 - acc: 0.9887 - val_loss: 0.0756 - val_acc: 0.9812
Epoch 8/20
60000/60000 [==============================] - 8s - loss: 0.0341 - acc: 0.9901 - val_loss: 0.0771 - val_acc: 0.9827
Epoch 9/20
60000/60000 [==============================] - 8s - loss: 0.0321 - acc: 0.9907 - val_loss: 0.0900 - val_acc: 0.9809
Epoch 10/20
60000/60000 [==============================] - 8s - loss: 0.0325 - acc: 0.9915 - val_loss: 0.0875 - val_acc: 0.9818
Epoch 11/20
60000/60000 [==============================] - 8s - loss: 0.0285 - acc: 0.9917 - val_loss: 0.0849 - val_acc: 0.9837
Epoch 12/20
60000/60000 [==============================] - 8s - loss: 0.0261 - acc: 0.9925 - val_loss: 0.0886 - val_acc: 0.9835
Epoch 13/20
60000/60000 [==============================] - 8s - loss: 0.0240 - acc: 0.9930 - val_loss: 0.1016 - val_acc: 0.9810
Epoch 14/20
60000/60000 [==============================] - 8s - loss: 0.0244 - acc: 0.9936 - val_loss: 0.0956 - val_acc: 0.9826
Epoch 15/20
60000/60000 [==============================] - 8s - loss: 0.0194 - acc: 0.9944 - val_loss: 0.0950 - val_acc: 0.9843
Epoch 16/20
60000/60000 [==============================] - 8s - loss: 0.0219 - acc: 0.9943 - val_loss: 0.1143 - val_acc: 0.9810
Epoch 17/20
60000/60000 [==============================] - 8s - loss: 0.0197 - acc: 0.9944 - val_loss: 0.1056 - val_acc: 0.9841
Epoch 18/20
60000/60000 [==============================] - 8s - loss: 0.0212 - acc: 0.9948 - val_loss: 0.1143 - val_acc: 0.9833
Epoch 19/20
60000/60000 [==============================] - 8s - loss: 0.0202 - acc: 0.9951 - val_loss: 0.1056 - val_acc: 0.9835
Epoch 20/20
60000/60000 [==============================] - 8s - loss: 0.0188 - acc: 0.9954 - val_loss: 0.1045 - val_acc: 0.9847
-------evaluate--------
 9952/10000 [============================>.] - ETA: 0sTest score: 0.104524913335
Test accuracy: 0.9847
```
