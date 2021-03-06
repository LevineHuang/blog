---
title: 基于TensorFlow实现多层感知器
date: 2017-02-28 12:17:00
updated	: 2018-06-27 12:17:00
permalink: abc
tags:
- Keras
- Deep Learning
- AI
- MI
categories:
- DeepLearning
- TensorFlow
---

## 基于TensorFlow实现多层感知器
----


```python
'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
```


```python
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/", one_hot=True)

import tensorflow as tf
```

    Extracting ../../data/train-images-idx3-ubyte.gz
    Extracting ../../data/train-labels-idx1-ubyte.gz
    Extracting ../../data/t10k-images-idx3-ubyte.gz
    Extracting ../../data/t10k-labels-idx1-ubyte.gz


```python
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
```


```python
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
```


```python
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
```


```python
# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
import tensorflow as tf
import numpy as np
A = [[1,3,4,5,6]]
B = [[1,3,4,3,2]]
with tf.Session() as sess:
    print(sess.run(tf.equal(A, B)))
```

    [[ True  True  True False False]]


```python
# tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，
# 如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
import tensorflow as tf  
import numpy as np  
A = [[1,3,4,5,6]]  
B = [[1,3,4], [2,4,1]]  
with tf.Session() as sess:  
    print(sess.run(tf.argmax(A, 1)))  
    print(sess.run(tf.argmax(B, 1)))

# tf.cast：用于改变某个张量的数据类型

A = tf.convert_to_tensor(np.array([[1,1,2,4], [3,4,8,5]]))  

with tf.Session() as sess:  
    print(A.dtype)
    b = tf.cast(A, tf.float32)  
    print(b.dtype)
```

    [4]
    [2 1]
    <dtype: 'int64'>
    <dtype: 'float32'>


```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```

    Epoch: 0001 cost= 201.333574999
    Epoch: 0002 cost= 46.010647814
    Epoch: 0003 cost= 28.918377875
    Epoch: 0004 cost= 20.020559532
    Epoch: 0005 cost= 14.741128482
    Epoch: 0006 cost= 11.113235143
    Epoch: 0007 cost= 8.444069761
    Epoch: 0008 cost= 6.217341204
    Epoch: 0009 cost= 4.772409531
    Epoch: 0010 cost= 3.578625235
    Epoch: 0011 cost= 2.738378037
    Epoch: 0012 cost= 1.975170798
    Epoch: 0013 cost= 1.481307366
    Epoch: 0014 cost= 1.183054283
    Epoch: 0015 cost= 0.933368129
    Optimization Finished!
    Accuracy: 0.9483
