### 1.构建模型，可视化网络结构

- Sequential add 

- Model API构建复杂网络结构

- 网络结构可视化

  ```
  
  ```

- 网格搜索调参

  1. 先通过入参定义模型结构，将定义的模型用keras中的wrapper进行包装，然后调用sklearn中的GridSearchCV，并输入需要搜索的参数列表param_grid以及评估方法。
  2. 输入训练数据，进行模型拟合，搜索最后参数组合。
  3. 获取最优模型 best_model = validator.best_estimator_.model。

  ```python
  from __future__ import print_function
  
  import keras
  from keras.datasets import mnist
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Activation, Flatten
  from keras.layers import Conv2D, MaxPooling2D
  from keras.wrappers.scikit_learn import KerasClassifier
  from keras import backend as K
  from sklearn.grid_search import GridSearchCV
  
  
  num_classes = 10
  
  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)
  
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  
  def make_model(dense_layer_sizes, filters, kernel_size, pool_size):
      model = Sequential()
      model.add(Conv2D(filters, kernel_size,
                       padding='valid',
                       input_shape=input_shape))
      model.add(Activation('relu'))
      model.add(Conv2D(filters, kernel_size))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=pool_size))
      model.add(Dropout(0.25))
  
      model.add(Flatten())
      for layer_size in dense_layer_sizes:
          model.add(Dense(layer_size))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))
      model.add(Dense(num_classes))
      model.add(Activation('softmax'))
  
      model.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
  
      return model
  
  dense_size_candidates = [[32], [64], [32, 32], [64, 64], [32, 64], [64, 32]]
  my_classifier = KerasClassifier(make_model, batch_size=32)
  validator = GridSearchCV(my_classifier,
                           param_grid={'dense_layer_sizes': dense_size_candidates,
                                       'epochs': [3, 6],
                                       'filters': [8],
                                       'kernel_size': [3],
                                       'pool_size': [2]},
                           scoring='log_loss',
                           n_jobs=1)
  
  validator.fit(x_train, y_train)
  
  # result:
  # The parameters of the best model are:
  # {'dense_layer_sizes': [64, 64], 'pool_size': 2, 'filters': 8, 'epochs': 6, 'kernel_size': 3}
  print('The parameters of the best model are: ')
  print(validator.best_params_)
  
  best_model = validator.best_estimator_.model
  metric_names = best_model.metrics_names
  metric_values = best_model.evaluate(x_test, y_test)
  for metric, value in zip(metric_names, metric_values):
      print(metric, ': ', value)
  ```

  

### 2.compile

### 3.fit

### 4.evaluate

### 5.predict

### 6.借助Tensorboard训练过程的监控

### 7.模型保存、载入、定期保存checkpoint

```python
# save model to JSON
model_json = model.to_json()
with open("SaveModel/cifarCnnModelnew.json", "w") as json_file:
    json_file.write(model_json)
    
 # save model to yaml
model_yaml = model.to_yaml()
with open("SaveModel/cifarCnnModelnew.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# Save Weight to h5
model.save_weights("SaveModel/cifarCnnModelnew.h5")
print("Saved model to disk")

# load trained model
try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")

```





