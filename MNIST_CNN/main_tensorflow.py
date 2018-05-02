import numpy as np
import tensorflow as tf

## tensorflow 安装 -------------------------------------------------------------
'''
建议采用conda环境来管理python以及组件的版本。
https://conda.io/miniconda.html

tensorflow目前支持linux, mac, windows平台，安装方法见官网。
https://www.tensorflow.org/install/

例如，对于windows平台+conda环境+python3.x+CPU，安装方法为：
pip install --ignore-installed --upgrade tensorflow
'''

## 设置日志级别 -----------------------------------------------------------------
'''
TensorFlow用五个不同级别的日志信息。为了升序的严重性，他们是调试DEBUG，信息INFO，警告WARN，
错误ERROR和致命FATAL的。当你配置日志记录在任何级别，TensorFlow将输出对应于更高程度的严重性
和所有级别的日志信息。例如，如果设置错误的日志记录级别，将得到包含错误和致命消息的日志输出，
并且如果设置了调试级别，则将从所有五个级别获取日志消息。

默认情况下，TensorFlow配置在日志记录级别的WARN，但当跟踪模型的训练，你会想要调整水平到INFO，
这将提供额外的反馈如进程中的fit操作。
'''
tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.WARN)


## main 函数 -------------------------------------------------------------------
'''
注意：tensorflow的典型用法是使用tf.app.run()来运行全部内容（见本脚本最后）。改命令需要我们
实现一个main(unused_argv)函数。

https://www.tensorflow.org/tutorials/layers
'''
def main(unused_argv):
    ## 数据读取 -----------------------------------------------------------------
    '''
    tf.contrib.learn.datasets.load_dataset提供了读取mnist数据的方式， 改方法可能会被
    官方废弃， 所以会在运行时打印一堆警告信息，暂不影响使用。
    '''
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print('train_data   :', train_data.shape)
    print('train_labels :', train_labels.shape)
    print('eval_data    :', eval_data.shape)
    print('eval_labels  :', eval_labels.shape)

    # 初始化 Estimator ----------------------------------------------------------
    '''
    Estimator（评估器）类代表一个模型，以及这些模型被训练和评估的方式。
    https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator

    TensorFlow 中有许多流行的库，如 Keras、TFLearn 和 Sonnet，它们可以让你轻松训练模型，
    而无需接触哪些低级别函数。同理，tf.estimator是tensorflow的一个高级的机器学习的API，
    使训练，评估多种机器学习的模型更简单。 
    '''
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./mnist_convnet_model")

    # 如果需要在训练中打印tensor -------------------------------------------------
    '''
    https://www.tensorflow.org/api_docs/python/tf/train/LoggingTensorHook
    '''
    tensors_to_log = {"probabilities": "softmax_tensor"}
    tensors_to_log = {} # 为了简介，这里设置为空
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50)

    # 模型训练 ------------------------------------------------------------------
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=64,
        num_epochs=10,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # 模型测试 ------------------------------------------------------------------
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

# 定义Estimator的model_fn ------------------------------------------------------
'''
模型函数将输入特征作为参数，相应标签作为张量。它还有一种模式来标记模型是否正在训练、评估或
执行推理。模型函数的最后一个参数是超参数的集合，它们与传递给 Estimator 的内容相同。模型函数
需要返回一个 EstimatorSpec 对象——它会定义完整的模型。

例如，推理模式的返回：
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

EstimatorSpec 接受预测，损失，训练和评估几种操作，因此它定义了用于训练，评估和推理的完整模型图。
'''
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    tf.app.run()
