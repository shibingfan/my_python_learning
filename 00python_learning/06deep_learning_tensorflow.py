# #  TFRecord文件存储
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
#
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
# mnist = input_data.read_data_sets("MNIST_data", dtype=tf.uint8, one_hot=True)
# images = mnist.train.images
# labels = mnist.train.labels
#
# def critical(value):
#     for i in range(5):
#         if value[i] == 1:
#             return True
#     return False
#
# need_index = []
# for i, val in enumerate(labels):
#     if critical(val):
#         need_index.append(i)
# pixels = images.shape[1]
# num_examples = int(mnist.train.num_examples)
#
# filename = "train.tfrecords"
# writer = tf.python_io.TFRecordWriter(filename)
# for index in need_index:
#     image_raw = images[index].tostring()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         # 'pixels':_int64_featue(pixels),
#         'label': _int64_feature(np.argmax(labels[index])),
#         'img_raw': _bytes_feature(image_raw)}))  # example对象对label和image数据进行封装
#     writer.write(example.SerializeToString())
# writer.close()
# print("TFRecord格式训练文件已保存")
#
# import os
#
#
# # 解析一个TFRecord样本
# def parser(record):
#     queue = tf.train.string_input_producer([record])  # 根据TFRecord文件生成队列
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(queue)
#     features = tf.parse_single_example(serialized_example, features={
#         'img_raw': tf.FixedLenFeature([], tf.string),
#         'label': tf.FixedLenFeature([], tf.int64)
#     })
#     # tf.decode_raw() 用于将numpy array 解析成图像对应的像素数组
#     decoded_image = tf.decode_raw(features['img_raw'], tf.uint8)
#     image = tf.reshape(decoded_image, [784])
#     image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # 将像素值归一化到[-0.5, 0.5]
#     label = tf.cast(features['label'], tf.int64)
#     return image, label
#
#
# # 定义单隐层神经网络的前向传播过程，并返回神经网络的前向传播结果
# def inference(input_tensor, weights1, biases1, weights2, biases2):
#     layer = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
#     return tf.matmul(layer, weights2) + biases2
#
#
# image, label = parser("train.tfrecords")
# # 将处理后的图像和标签数据通过tf.train.shuffle_batch整理成ANN训练时所需要的batch
# img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=5, capacity=1015, min_after_dequeue=500)
#
# # 神经网络模型的相关参数设置
# INPUT_NODE = 784
# OUTPUT_NODE = 5
# LAYER1_NODE = 500  # 隐层神经元节点数
# LEARNING_RATE = 0.002  # 初始学习率设定
# REGULARAZTION_RATE = 0.0001  # 正则化系数设定
# TRAINING_STEPS = 20000  # 总训练步数
# current_path = r'C:/Users/Lenovo/Desktop/data/'
# MODEL_PATH = "model/"  # 定义模型保存地址与文件名
# MODEL_NAME = "EXAMPLE.ckpt"
#
# weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
# biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
#
# weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
# biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
#
# y = inference(img_batch, weights1, biases1, weights2, biases2)
#
# # 计算交叉熵及其在当前训练batch上的平均值
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
# cross_entropy_mean = tf.reduce_mean(cross_entropy)
#
# # 损失函数的计算，包含两项，其一：平均交叉熵损失，其二：正则化项
# regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
# regularaztion = regularizer(weights1) + regularizer(weights2)
# loss = cross_entropy_mean + regularaztion
#
# # 优化损失函数，这里使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
#
# # 准确率计算，先将精度转换为float32，再计算准确率
# prediction = tf.equal(tf.argmax(y, 1), label_batch)  # label_batch类似[1, 4, 1, 2, 0]
# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
#
# # 初始化TensorFlow持久化类，用于模型的保存
# saver = tf.train.Saver()
# with tf.Session() as sess:  # 创建会话
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     coord = tf.train.Coordinator()  # 用于完成多线程协同的功能，该类实例需要明确调用下述语句来启动所有线程
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for i in range(1, TRAINING_STEPS + 1):
#         sess.run(train_step)
#         if i % 2000 == 0:
#             print("After {} training step(s), loss is {}, and accuracy is {}".format(i, sess.run(loss),
#                                                                                      sess.run(accuracy)))
#             saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME))
#     coord.request_stop()
#     coord.join(threads)  # 用于停止所有进程


# # TensorFlow——基于BP模型和MNIST数据集的手写数字识别
# import tensorflow as tf
# # 调用Tensorflow中MNIST(手写数字）数据集
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
#
# # 定义占位符，转换长784向量为28*28*1
# with tf.variable_scope('Placeholder'):
#     x = tf.placeholder("float", name='X_placeholder', shape=[None, 784])
#     y_ = tf.placeholder("float", name='y_placeholder', shape=[None, 10])
#     x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# # 两个卷积层、两个池化层、一个全连接层和一个softmax层。Softmax层用于分类
# with tf.variable_scope('conv1') as scope:
#     W_conv1 = tf.get_variable('W_conv1', shape=[5, 5, 1, 32], initializer=tf.random_normal_initializer(stddev=0.1))
#     b_conv1 = tf.get_variable('b_conv1', shape = [32], initializer=tf.constant_initializer(0.1))
#     conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
#     pre_activation = tf.nn.bias_add(conv1, b_conv1)
#     activation = tf.nn.relu(pre_activation, name=scope.name)
#
# with tf.variable_scope('pool1') as scope:
#     pool1 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name)
#
# with tf.variable_scope('conv2') as scope:
#     W_conv2 = tf.get_variable('W_conv2', shape=[5, 5, 32, 64], initializer=tf.random_normal_initializer(stddev=0.1))
#     b_conv2 = tf.get_variable('b_conv2', shape=[64], initializer=tf.constant_initializer(0.1))
#     conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
#     pre_activation = tf.nn.bias_add(conv2, b_conv2)
#     activation = tf.nn.relu(pre_activation, name=scope.name)
#
# with tf.variable_scope('pool2') as scope:
#     pool2 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name)
#
# with tf.variable_scope('fc1') as scope:
#     W_fc1 = tf.get_variable("W_fc1", shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
#     b_fc1 = tf.get_variable("b_fc1", shape=[1024], initializer=tf.constant_initializer(0.1))
#     pool2_falt = tf.reshape(pool2, [-1, 7 * 7 * 64])
#     fc1 = tf.matmul(pool2_falt, W_fc1) + b_fc1
#     activation = tf.nn.relu(fc1, name=scope.name)
#     drop_fc1 = tf.nn.dropout(activation, keep_prob=0.5)
#
# with tf.variable_scope('softmax') as scope:
#     W_softmax = tf.get_variable("W_softmax", shape=[1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
#     b_softmax = tf.get_variable("b_softmax", shape=[10], initializer=tf.constant_initializer(0.1))
#     y_conv = tf.nn.softmax(tf.matmul(drop_fc1, W_softmax) + b_softmax, name=scope.name)
#
# # 损失函数是目标类别和预测类别之间的交叉熵
# with tf.variable_scope('Loss'):
#     cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
# # 计算准确率
# '''tf.argmax()函数：是返回对象在某一维上的其数据最大值所对应的索引值，由于这里的标签向量都是由0,1组成，因此最大值1所在的索引位置就是对应的类别标签
# tf.argmax(y_conv,1)返回的是对于任一输入x预测到的标签值，tf.argmax(y_,1)代表正确的标签值
# correct_prediction 返回一个布尔数组。为计算分类准确率，将布尔值转换为浮点数来代表对与错，然后取平均值。如[True, False, True, True]变为[1,0,1,1]，计算出准确率就为0.75。'''
# with tf.variable_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     # batch是从MNIST数据集中按批次数取得的：数据项与标签项feed_dict=({x:batch[0],y_:batch[1]}语句：是将batch[0]、batch[1]代表的值传入x、y_。
#     for i in range(20000):
#         batch = mnist.train.next_batch(50)
#         if i % 1000 == 0:
#             train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
#             print("step %d, training accuracy %g" % (i, train_accuracy))
#         train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
#     print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#     # 模型保存
#     saver = tf.train.Saver()
#     last_chkp = saver.save(sess, 'results/graph.chkp')
#     # sv.saver.save(sess, 'results/graph.chkp')
#
# for op in tf.get_default_graph().get_operations():
#     print(op.name)

# # TensorFlow——基于LeNet模型和MNIST数据集的手写数字识别
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# sess = tf.InteractiveSession()
#
# x = tf.placeholder("float", shape=[None, 784])
# y_ = tf.placeholder("float", shape=[None, 10])
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#
#
# # 将图片变为4d张量，28*28，第四维对应图片通道数，灰度为1，彩色为3
# x_image = tf.reshape(x, [-1, 28, 28, 1])
# # 第一层卷积，卷积核5*5，输入通道1，输出通道32个特征图。
# # 将x_image与权重张量进行卷积，加偏置项，使用 ReLU激活函数，最后max_pool_2x2最大池化操作将图片减小到14*14。
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
# # 第二层卷积
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
# # 全连接层，1024个神经元全连接，用于处理整张图像
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# # dropout
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# # 输出层
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# # 训练评估
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   # 交叉熵损失
# # 定义优化策略：采用Adam优化算法，以0.0001的学习率进行最小化损失操作
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# # 判断预测值与标签是否相等，返回一个布尔值
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 将布尔值转化为数值，计算平均准确率
# sess.run(tf.initialize_all_variables())  # 变量初始化
# for i in range(2000):             # 在循环中实现参数更新
#     batch = mnist.train.next_batch(50)  #batch size为50，每次获取50张数据
#     if i%500 == 0:              # 500次在屏幕打印一次信息
#         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})   # 将数据填入占位符，喂给网络
#         print("step %d, training accuracy %g"%(i, train_accuracy))  # 输出迭代次数和测试训练的准确率
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# # TensorFlow——单层感知机和多层感知机的实现
# # 单层感知机
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
#
#
# class TrainDataLoader:
#     def __init__(self):
#         pass
#     def GenerateRandomData(self, count, grandient, offset):
#         x1 = np.linspace(1, 5, count)
#         x2 = grandient * x1 + np.random.randint(-10, 10, *x1.shape) + offset
#         dataset = []
#         y = []
#         for i in range(*x1.shape):
#             dataset.append([x1[i], x2[i]])
#             real_value = grandient * x1[i] + offset
#             if real_value > x2[i]:
#                 y.append(-1)
#             else:
#                 y.append(1)
#         return x1, x2, np.mat(y), np.mat(dataset)
#
#
# class SimplePerceptron:
#     def __init__(self, train_data=[], real_result=[], eta=1):
#         self.w = np.zeros([1, len(train_data.T)], int)
#         self.b = 0
#         self.eta = eta
#         self.train_data = train_data
#         self.real_result = real_result
#
#     def nomalize(self, x):
#         if x > 0:
#             return 1
#         else:
#             return -1
#
#     def model(self, x):
#         # Here are matrix dot multiply get one value
#         y = np.dot(x, self.w.T) + self.b
#         # Use sign to nomalize the result
#         predict_v = self.nomalize(y)
#         return predict_v, y
#
#     def update(self, x, y):
#         # w = w + n*y_i*x_i
#         self.w = self.w + self.eta * y * x
#         # b = b + n*y_i
#         self.b = self.b + self.eta * y
#
#     def loss(slef, fx, y):
#         return fx.astype(int) * y
#
#     def train(self, count):
#         update_count = 0
#         while count > 0:
#             # count--
#             count = count - 1
#             if len(self.train_data) <= 0:
#                 print("exception exit")
#                 break
#             # random select one train data
#             index = np.random.randint(0, len(self.train_data) - 1)
#             x = self.train_data[index]
#             y = self.real_result.T[index]
#             # wx+b
#             predict_v, linear_y_v = self.model(x)
#             # y_i*(wx+b) > 0, the classify is correct, else it's error
#             if self.loss(y, linear_y_v) > 0:
#                 continue
#             update_count = update_count + 1
#             self.update(x, y)
#         print("update count: ", update_count)
#         pass
#
#     def verify(self, verify_data, verify_result):
#         size = len(verify_data)
#         failed_count = 0
#         if size <= 0:
#             pass
#         for i in range(size):
#             x = verify_data[i]
#             y = verify_result.T[i]
#             if self.loss(y, self.model(x)[1]) > 0:
#                 continue
#             failed_count = failed_count + 1
#         success_rate = (1.0 - (float(failed_count) / size)) * 100
#         print("Success Rate: ", success_rate, "%")
#         print("All input: ", size, " failed_count: ", failed_count)
#
#     def predict(self, predict_data):
#         size = len(predict_data)
#         result = []
#         if size <= 0:
#             pass
#         for i in range(size):
#             x = verify_data[i]
#             y = verify_result.T[i]
#             result.append(self.model(x)[0])
#         return result
#
# if __name__ == "__main__":
#     # Init some parameters
#     gradient = 2
#     offset   = 10
#     point_num = 1000
#     train_num = 50000
#     loader = TrainDataLoader()
#     x, y, result, train_data =  loader.GenerateRandomData(point_num, gradient, offset)
#     x_t, y_t, test_real_result, test_data =  loader.GenerateRandomData(100, gradient, offset)
#
#     # First training
#     perceptron = SimplePerceptron(train_data, result)
#     perceptron.train(train_num)
#     perceptron.verify(test_data, test_real_result)
#     print("T1: w:", perceptron.w," b:", perceptron.b)
#
#     # Draw the figure
#     # 1. draw the (x,y) points
#     plt.plot(x, y, "*", color='gray')
#     plt.plot(x_t, y_t, "+")
#     # 2. draw y=gradient*x+offset line
#     plt.plot(x,x.dot(gradient)+offset, color="red")
#     # 3. draw the line w_1*x_1 + w_2*x_2 + b = 0
#     plt.plot(x, -(x.dot(float(perceptron.w.T[0]))+float(perceptron.b))/float(perceptron.w.T[1]), color='green')
#     plt.show()
#     plt.savefig('result.png')


# # 多层感知机
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib
# matplotlib.use('Agg')
#
# # 0，导入数据
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images.shape, mnist.train.labels.shape)
#
# # 1，定义模型计算公式
# sess = tf.InteractiveSession()
# in_units = 784
# h1_units = 300
# W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
# b1 = tf.Variable(tf.zeros([h1_units]))
# W2 = tf.Variable(tf.zeros([h1_units, 10]))
# b2 = tf.Variable(tf.zeros([10]))
#
# X = tf.placeholder(tf.float32, [None, in_units])
# keep_prob = tf.placeholder(tf.float32, )
# h1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# h1_drop = tf.nn.dropout(h1, keep_prob)
# y_pred = tf.nn.softmax(tf.matmul(h1_drop, W2) + b2)
#
# # 2,定义loss，选定优化器
# y = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
# train_step = tf.train.AdagradOptimizer(learning_rate=0.3).minimize(cross_entropy)
#
# # 3,定义精确度计算公式
# correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # 4,初始化参数
# tf.global_variables_initializer().run()
#
# # 4,迭代训练
# epoch_num = 1
# for epoch in range(epoch_num):
#     avg_accuracy = 0.0
#     avg_cost = 0.0
#     for i in range(3000):
#         batch_xs, batch_ys = mnist.train.next_batch(100)
#         cost, acc, _ = sess.run([cross_entropy, accuracy, train_step],
#                                 feed_dict={X: batch_xs, y: batch_ys, keep_prob: 0.75})
#         avg_cost += cost
#         avg_accuracy += acc / 3000
#     print('Epoch %d: cost is %.7f,accuracy is %.7f.' % (epoch + 1, avg_cost, avg_accuracy))
# print('Train Finished!')
# print('Test accuracy is %.4f.' % accuracy.eval({X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
#
# # 5,Get one and predict
# import matplotlib.pyplot as plt
# import random
#
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction:", sess.run(tf.argmax(y_pred, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1.0}))
# plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
# # plt.show()
# plt.savefig('result1.png')
#
# sess.close()


# # TensorFlow——基于DNN模型和Iris data set的鸢尾花品种识别
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import os
# from six.moves.urllib.request import urlopen
# import tensorflow as tf
# import numpy as np
#
# IRIS_TRAINING = "iris_training.csv"
# IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
# IRIS_TEST = "iris_test.csv"
# IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
#
#
# def main():
#     if not os.path.exists(IRIS_TRAINING):
#         raw = urlopen(IRIS_TRAINING_URL).read()
#         with open(IRIS_TRAINING, 'wb') as f:
#             f.write(raw)
#
#     if not os.path.exists(IRIS_TEST):
#         raw = urlopen(IRIS_TEST_URL).read()
#         with open(IRIS_TEST, 'wb') as f:
#             f.write(raw)
#
#     # Load datasets.
#     training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)
#     test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)
#
#     # Specify that all features have real-value data
#     feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
#
#     # Build 3 layer DNN with 10, 20, 10 units respectively.
#     classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="/iris_model")
#
#     # Define the training inputs
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(training_set.data)}, y=np.array(training_set.target), num_epochs=None, shuffle=True)
#
#     # Train model.
#     classifier.train(input_fn=train_input_fn, steps=2000)
#
#     # Define the test inputs
#     test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(test_set.data)}, y=np.array(test_set.target), num_epochs=1, shuffle=False)
#
#     # Evaluate accuracy.
#     accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
#     print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
#
#     # Classify two new flower samples.
#     new_samples = np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
#     predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": new_samples}, num_epochs=1, shuffle=False)
#
#     predictions = list(classifier.predict(input_fn=predict_input_fn))
#     predicted_classes = [p["classes"] for p in predictions]
#     print("New Samples, Class Predictions:    {}\n".format(predicted_classes))
#
#
# if __name__ == "__main__":
#     main()


# TensorFlow——基于Time Series的时间序列预测
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")


x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 50) + np.cos(np.pi * x / 50) + np.sin(np.pi * x / 25) + noise

plt.plot(x, y, 'g-', label='Input data')
plt.legend()
# plt.savefig('Input_timeseries_data.jpg')

data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
}

# reader = tf.contrib.timeseries.CSVReader(csv_file_name)
reader = NumpyReader(data)

with tf.Session() as sess:
    full_data = reader.read_full()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print(sess.run(full_data))
    coord.request_stop()

train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=2, window_size=10)
with tf.Session() as sess:
    batch_data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    one_batch = sess.run(batch_data[0])
    coord.request_stop()
print('one_batch_data:', one_batch)



