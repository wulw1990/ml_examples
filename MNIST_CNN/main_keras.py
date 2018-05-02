## keras 简介和安装 -------------------------------------------------------------
'''
Keras简介：
    Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK后端。
    也就是说，Keras是对其他库的封装，所以需要先安装一个后端，推荐用Tensorflow作为后端。
    Keras存在的目的是简化原型设计，原因是Tensorflow、Theano以及CNTK对新手不够友好。
    但是，如果使用PyTorch则不存在这个问题，因为PyTorch本身设计就已经很简洁。
    PyTorch和keras一样的易于上手，但是避免了keras对具体细节的屏蔽导致的调试和研究的困难。

安装Conda:
    建议采用conda环境来管理python以及组件的版本。
    https://conda.io/miniconda.html

安装TensorFlow:
    tensorflow目前支持linux, mac, windows平台，安装方法见官网。
    https://www.tensorflow.org/install/

    对于windows平台+conda环境+python3.x+CPU，安装方法为：
    pip install --ignore-installed --upgrade tensorflow
    
    对于linux需要手动指定URL:
    https://www.tensorflow.org/install/install_linux#InstallingAnaconda

安装Keras:
    完整安装可以参考新手指南，比如：
    http://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/
    
    对于跑通我们这个示例，只需要执行：
    pip install keras
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K



# input image dimensions

## 初始化Dataset ----------------------------------------------------------------
'''
keras.datasets提供的mnist模块非常方便，他会在首次使用时自动下载数据集。
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10
img_rows, img_cols = 28, 28

## 数据格式重组 ------------------------------------------------------------------
'''
由于Keras可能依赖不同的后端，而不同后端的数据格式有差异,主要是差异为是否channels_first
对于channels_first的输入格式为: [num, channels, rows, cols]
反之为：[num, rows, cols， channels]
TensorFlow使用的是后者。
'''
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

## 数据预处理 ------------------------------------------------------------------
'''
这里对数据做了简单的归一化，除以255，是的数值范围变成[0, 1.0]之间。
另外，对数据标签(label)做了one-hot处理。比如label=3, one-hot后变成[0 0 0 1 0 0 0 0 0 0]
即index为3的位置为1，其他位置为0。得到的向量长度为10，因为mnist一共有10类数字。
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## 定义模型的网络结构 -------------------------------------------------------------
model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

## 编译网络 ---------------------------------------------------------------------
'''
注意： Keras网络定义后，需要compile方可使用。编译的时候也指定loss，optimizer，metrics。
'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

## 训练 ------------------------------------------------------------------------
'''
fit函数，可以设置validation_data，这样可以边训练边测试验证集（这里直接用的测试集）的精度。
https://keras-cn.readthedocs.io/en/latest/models/model/
'''
model.fit(x=x_train,
          y=y_train,
          batch_size=256,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))

## 测试 ------------------------------------------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## 预测 ------------------------------------------------------------------------
# y_test_predict = model.predict(x_test)
