# # Keras——Dropout
# # 输入层使用dropout
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dropout
# from keras.layers import Dense
# from keras.optimizers import SGD
# from keras.wrappers.scikit_learn import KerasClassifier
# from python_packages.model_selection import cross_val_score
# from python_packages.model_selection import KFold
#
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 构建模型函数
# def create_model(init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     # 输入层使用dropout
#     model.add(Dropout(rate=0.2, input_shape=(4,)))
#     model.add(Dense(units=4, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     '''
#     model = Sequnential()
#     隐藏层使用dropout
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init, kernel_constraint=maxnorm(3)))
#     model.add(Dropout(rate=0.2))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init, kernel_constraint=maxnorm(3)))
#     model.add(Dropout(rate=0.2))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     '''
#     # 定义Dropout
#     sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model
# model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=1)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(model, x, Y, cv=kfold)
# print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))


# # Keras——学习率衰减
# # 学习率线性衰减
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.optimizers import SGD
#
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 构建模型函数
# def create_model(init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     #模型优化
#     learningRate = 0.1
#     momentum = 0.9
#     decay_rate = 0.005
#     sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model
# epochs = 200
# model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=5, verbose=1)
# model.fit(x, Y)
# # 指数衰减
# # 学习率指数衰减
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.optimizers import SGD
# from keras.callbacks import LearningRateScheduler
# from math import pow, floor
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 计算学习率
# def step_decay(epoch):
#     init_lrate = 0.1
#     drop = 0.5
#     epochs_drop = 10
#     lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
#     return lrate
# # 构建模型函数
# def create_model(init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     #模型优化
#     learningRate = 0.1
#     momentum = 0.9
#     decay_rate = 0.0
#     sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model
# lrate = LearningRateScheduler(step_decay)
# epochs = 200
# model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=5, verbose=1, callbacks=[lrate])
# model.fit(x, Y)


# # Keras——基于多变量时间序列的PM2.5预测
# from pandas import DataFrame, concat, read_csv
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.optimizers import SGD
# from python_packages.preprocessing import LabelEncoder, MinMaxScaler
# from matplotlib import pyplot as plt
# from datetime import datetime
# import matplotlib
# matplotlib.use('Agg')
#
# batch_size = 72
# epochs = 50
# # 通过过去几次的数据进行预测
# n_input = 1
# n_train_hours = 365 * 24 * 4
# n_validation_hours = 24 * 5
#
# current_path = r'C:/Users/Lenovo/Desktop/data/'
# file_path = r'data/pollution_original.csv'
# filename = current_path + file_path
#
# def prase(x):
#     return datetime.strptime(x, '%Y %m %d %H')
#
# def load_dataset():
#     # 导入数据
#     dataset = read_csv(filename, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=prase)
#     # 删除No.列
#     dataset.drop('No', axis=1, inplace=True)
#     # 设定列名
#     dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#     dataset.index.name = 'date'
#     # 使用中位数填充缺失值
#     dataset['pollution'].fillna(dataset['pollution'].mean(), inplace=True)
#     return dataset
#
# def convert_dataset(data, n_input=1, out_index=0, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = [], []
#     # 输入序列 (t-n, ... t-1)
#     for i in range(n_input, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # 输出结果 (t)
#     cols.append(df[df.columns[out_index]])
#     names += ['result']
#     # 合并输入输出序列
#     result = concat(cols, axis=1)
#     result.columns = names
#     # 删除包含缺失值的行
#     if dropnan:
#         result.dropna(inplace=True)
#     return result
#
# # class_indexs 编码的字段序列号，或者序列号List，列号从0开始
# def class_encode(data, class_indexs):
#     encoder = LabelEncoder()
#     class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]
#     values = DataFrame(data).values
#     for index in class_indexs:
#         values[:, index] = encoder.fit_transform(values[:, index])
#     return DataFrame(values) if type(data) is DataFrame else values
#
# def build_model(lstm_input_shape):
#     model = Sequential()
#     model.add(LSTM(units=50, input_shape=lstm_input_shape, return_sequences=True))
#     model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(1))
#     model.compile(loss='mae', optimizer='adam')
#     model.summary()
#     return model
#
# if __name__ == '__main__':
#     # 导入数据
#     data = load_dataset()
#     # 对风向列进行编码
#     data = class_encode(data, 4)
#     # 生成数据集，使用前5次的数据，来预测新数据
#     dataset = convert_dataset(data, n_input=n_input)
#     values = dataset.values.astype('float32')
#     # 分类训练与评估数据集
#     train = values[:n_train_hours, :]
#     validation = values[-n_validation_hours:, :]
#     x_train, y_train = train[:, :-1], train[:, -1]
#     x_validation, y_validation = validation[:, :-1], validation[:, -1]
#     # 数据归一元(0-1之间)
#     scaler = MinMaxScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_validation = scaler.fit_transform(x_validation)
#     # 将数据整理成【样本，时间步长，特征】结构
#     x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
#     x_validation = x_validation.reshape(x_validation.shape[0], 1, x_validation.shape[1])
#     # 查看数据维度
#     print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
#     # 训练模型
#     lstm_input_shape = (x_train.shape[1], x_train.shape[2])
#     model = build_model(lstm_input_shape)
#     model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_validation, y_validation), epochs=epochs,
#               verbose=2)
#     # 使用模型预测评估数据集
#     prediction = model.predict(x_validation)
#     # 图表显示
#     plt.plot(y_validation, color='blue', label='Actual')
#     plt.plot(prediction, color='green', label='Prediction')
#     plt.legend(loc='upper right')
#     plt.show()
#     plt.savefig('PM2.5.png')


# # Keras——基于多层感知器的印第安人糖尿病诊断
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy as np
#
# # 设定随机数种子，使用固定随机种子初始化随机数生成器，这样就可以重复地运行相同的代码，并获得相同的结果
# np.random.seed(7)
# # 导入数据，使用np.loadtxt
# current_path = r'C:/Users/Lenovo/Desktop/data/'
# dataset = np.loadtxt(current_path+'data/pima-indians-diabetes.csv', delimiter=',')
# # 分割输入变量x和输出变量y
# x = dataset[:, 0 : 8]
# Y = dataset[:, 8]
#
# # 创建模型，首先要确保输入层具有正确的输入维度。神经元数量（unit）初始化方法（init），激活函数activation
# # 使用三层连接的网络结构，通过Sequential的add函数将层添加到模型，并组合在一起
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))   # 第一个隐藏层有12个神经元，使用8个输入变量
# model.add(Dense(8, activation='relu'))               # 第二层隐藏层有8个神经元
# model.add(Dense(1, activation='sigmoid'))            # 输出层有一个神经元来预测数据结果
#
# # 评估一组权重的损失函数（loss），用于搜索网络不同权重的优化器（optimizer）
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(x=x, y=Y, epochs=150, batch_size=10)
#
# scores = model.evaluate(x=x, y=Y)
# print('\n%s : %.2f%%' % (model.metrics_names[1], scores[1]*100))


# # Keras——基于JSON和YAML的模型序列化
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
# from keras.models import model_from_json, model_from_yaml
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# Y_labels = to_categorical(Y, num_classes=3)
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 构建模型函数
# def create_model(optimizer='rmsprop', init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
# # 构建模型
# model = create_model()
# model.fit(x, Y_labels, epochs=200, batch_size=5, verbose=1)
# scores = model.evaluate(x, Y_labels, verbose=0)
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
#
# current_path = r'C:/Users/Lenovo/Desktop/data/'
# # 模型保存成Json文件
# model_json = model.to_json()
# with open(current_path + 'models/model.json', 'w') as file:
#     file.write(model_json)
# # 保存模型的权重值
# model.save_weights(current_path+'models/model.json.h5')
#
# # 从Json加载模型
# with open(current_path+'models/model.json', 'r') as file:
#     model_json = file.read()
# # 加载模型
# new_model = model_from_json(model_json)
# new_model.load_weights(current_path+'models/model.json.h5')
# # 编译模型
# new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# # 评估从Json加载的模型
# scores = new_model.evaluate(x, Y_labels, verbose=0)
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
#
# # 模型保存成Yaml文件
# model_yaml = model.to_yaml()
# with open(current_path + 'models/model.yaml', 'w') as file:
#     file.write(model_yaml)
# # 保存模型的权重值
# model.save_weights(current_path + 'models/model.yaml.h5')
# # 从Json加载模型
# with open(current_path + 'models/model.yaml', 'r') as file:
#     model_json = file.read()
# # 加载模型
# new_model = model_from_yaml(model_json)
# new_model.load_weights(current_path + 'models/model.yaml.h5')
# # 编译模型
# new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# # 评估从YAML加载的模型
# scores = new_model.evaluate(x, Y_labels, verbose=0)
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))


# # Keras——基于CNN模型和鸢尾花数据集的分类
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import  Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from python_packages.model_selection import cross_val_score, KFold
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 构建模型函数
# def create_model(optimizer='adam', init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='relu', kernel_initializer=init))
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
# # 模型评估
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(model, x, Y, cv=kfold)
# print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))


# # Keras——基于CNN模型和CIFAR-10数据集的分类
# import numpy as np
# from keras.datasets import cifar10
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.optimizers import SGD
# from keras.constraints import maxnorm
# from keras.utils import np_utils
# from keras import backend
# backend.set_image_data_format('channels_first')
#
# # 设定随机种子
# seed = 7
# np.random.seed(seed=seed)
# # 导入数据
# (X_train, y_train), (X_validation, y_validation) = cifar10.load_data()
# # 格式化数据到0-1之间
# X_train = X_train.astype('float32') / 255.0
# X_validation = X_validation.astype('float32') / 255.0
# # one-hot编码
# y_train = np_utils.to_categorical(y_train)
# y_validation = np_utils.to_categorical(y_validation)
# num_classes = y_train.shape[1]
#
# # 简单卷积神经网络:两个卷积层、一个池化层、一个Flatten层和一个全连接层
# def create_model(epochs=25):
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#     lrate = 0.01
#     decay = lrate / epochs
#     sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model
# epochs = 25
# model = create_model(epochs)
# model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)
# scores = model.evaluate(x=X_validation, y=y_validation, verbose=0)
# print('Accuracy: %.2f%%' % (scores[1] * 100))
# # 大型卷积神经网络:按照特征图是32、63、128各两次重复构建模型。
# def create_model2(epochs=25):
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dropout(0.2))
#     model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Dense(10, activation='softmax'))
#     lrate = 0.01
#     decay = lrate / epochs
#     sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model
#
# model2 = create_model2(epochs)
# model2.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)
# scores = model2.evaluate(x=X_validation, y=y_validation, verbose=0)
# print('Accuracy: %.2f%%' % (scores[1] * 100))


# # Keras——基于CNN模型和MNIST数据集的手写数字识别
# import numpy as np
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers.convolutional import  Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.utils import np_utils
# from keras import backend
# backend.set_image_data_format('channels_first')
#
# # 从Keras导入Mnist数据集
# (X_train, y_train), (X_validation, y_validation) = mnist.load_data()
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# num_pixels = X_train.shape[1] * X_train.shape[2]
# print(num_pixels, X_train.shape, y_validation.shape)
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') / 255.0
# X_validation = X_validation.reshape(X_validation.shape[0], num_pixels).astype('float32') / 255.0
# # one-hot编码
# y_train = np_utils.to_categorical(y_train)
# y_validation = np_utils.to_categorical(y_validation)
# num_classes = y_validation.shape[1]
# print(num_classes)
#
# # 定义基准多层感知机MLP模型:输入层784，隐藏层784，输出层10
# def create_model():
#     # 创建模型
#     model = Sequential()
#     model.add(Dense(units=num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# model = create_model()
# model.fit(X_train, y_train, epochs=10, batch_size=200)
# score = model.evaluate(X_validation, y_validation)
# print('MLP: %.2f%%' % (score[1] * 100))
#
# # 定义简单卷积神经网络：输入784，卷积层32maps5*5,池化层2*2，dropout层，flatten层，全连接层128，输出层10
# (X_train, y_train), (X_validation, y_validation) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255.0
# X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32') / 255.0
# y_train = np_utils.to_categorical(y_train)
# y_validation = np_utils.to_categorical(y_validation)
# def create_model2():
#     model = Sequential()
#     model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(units=128, activation='relu'))
#     model.add(Dense(units=10, activation='softmax'))
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# model2 = create_model2()
# model2.fit(X_train, y_train, epochs=10, batch_size=200, verbose=2)
# score = model2.evaluate(X_validation, y_validation, verbose=0)
# print('CNN_Small: %.2f%%' % (score[1] * 100))


# # Keras——模型增量更新
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
# from keras.models import model_from_json
# from python_packages.model_selection import train_test_split
#
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# x_train, x_increment, Y_train, Y_increment = train_test_split(x, Y, test_size=0.2, random_state=seed)
# # 将标签转换成分类编码
# Y_train_labels = to_categorical(Y_train, num_classes=3)
# # 构建模型函数
# def create_model(optimizer='rmsprop', init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
# # 构建模型
# model = create_model()
# model.fit(x_train, Y_train_labels, epochs=10, batch_size=5, verbose=2)
# scores = model.evaluate(x_train, Y_train_labels, verbose=0)
# print('Base %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
#
# # 模型保存成Json文件
# current_path = r'C:/Users/Lenovo/Desktop/data/models/'
# model_json = model.to_json()
# with open(current_path+'model.increment.json', 'w') as file:
#     file.write(model_json)
# # 保存模型的权重值
# model.save_weights(current_path+'model.increment.json.h5')
#
# # 从Json加载模型
# with open(current_path+'model.increment.json', 'r') as file:
#     model_json = file.read()
# # 加载模型
# new_model = model_from_json(model_json)
# new_model.load_weights(current_path+'model.increment.json.h5')
# # 编译模型
# new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# # 增量训练模型
# # 将标签转换成分类编码
# Y_increment_labels = to_categorical(Y_increment, num_classes=3)
# new_model.fit(x_increment, Y_increment_labels, epochs=10, batch_size=5, verbose=2)
# scores = new_model.evaluate(x_increment, Y_increment_labels, verbose=0)
# print('Increment %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))


# # Keras——模型评估
# from keras.models import Sequential
# from keras.layers import Dense
# from python_packages.model_selection import train_test_split
# import numpy as np
# from python_packages.model_selection import StratifiedKFold
#
# seed = 7
# # 设定随机数种子
# np.random.seed(seed)
# # 导入数据
# current_path = r'C:/Users/Lenovo/Desktop/data/data/'
# dataset = np.loadtxt(current_path+'pima-indians-diabetes.csv', delimiter=',')
# # 分割输入x和输出Y
# x = dataset[:, 0: 8]
# Y = dataset[:, 8]
# # 分割数据集
# x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=0.2, random_state=seed)
# # 构建模型
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # 编译模型
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # 自动评估
# # 训练模型并自动评估模型:设置验证分割参数为0.2，使用20%的数据评估模型
# model.fit(x=x, y=Y, epochs=150, batch_size=10, validation_split=0.2)
#
# # 手动评估
# # 手动分离数据进行评估
# model.fit(x_train, Y_train, validation_data=(x_validation, Y_validation), epochs=150, batch_size=10)
# # k折交叉验证：将数据集分为k个子集，选择其中一个子集作为评估数据集，利用剩余的k-1个子集训练模型，重复直至每个子集都被用作过评估
# # 使用StratifiedKFold将数据分割成10个子集，并利用这10个子集创建和评估10个模型，且收集这10个模型的评估得分。
# kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
# cvscores = []
# for train, validation in kfold.split(x, Y):
#     # 创建模型
#     model = Sequential()
#     model.add(Dense(12, input_dim=8, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # 编译模型
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # 训练模型
#     model.fit(x[train], Y[train], epochs=150, batch_size=10, verbose=1)
#     # 评估模型
#     scores = model.evaluate(x[validation], Y[validation], verbose=0)
#     # 输出评估结果
#     print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
#     cvscores.append(scores[1] * 100)
# # 输出均值和标准差
# print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))


# # Keras——模型训练可视化
# from python_packages import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
# import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.use('Agg')
# # 导入数据
# dataset = datasets.load_iris()
# x = dataset.data
# Y = dataset.target
# # 将标签转换成分类编码
# Y_labels = to_categorical(Y, num_classes=3)
# # 设定随机种子
# seed = 7
# np.random.seed(seed)
# # 构建模型函数
# def create_model(optimizer='rmsprop', init='glorot_uniform'):
#     # 构建模型
#     model = Sequential()
#     model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
#     model.add(Dense(units=6, activation='relu', kernel_initializer=init))
#     model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
# # 构建模型
# model = create_model()
# history = model.fit(x, Y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=1)
# # 评估模型
# scores = model.evaluate(x, Y_labels, verbose=0)
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
#
# # Hisotry列表.history类对象包含两个属性，分别为训练轮数epoch和字典history(包含val_loss,val_acc,loss,acc四个key值)。
# print(history.history.keys())
# # accuracy的历史
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# current_path = r'C:/Users/Lenovo/Desktop/data/images/'
# plt.savefig(current_path+'accuracy.png')
# plt.close()
# # loss的历史
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# plt.savefig(current_path+'loss.png')


# # Keras——图像增强
# from keras.preprocessing.image import ImageDataGenerator
# from keras.datasets import mnist
# from keras import backend
# import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.use('Agg')
# backend.set_image_data_format('channels_first')
#
# # 从Keras导入Mnist数据集
# (X_train, y_train), (X_validation, y_validation) = mnist.load_data()
# # 显示9张手写数字的图片
# for i in range(0, 9):
#     plt.subplot(331 + i)
#     plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# plt.show()
# plt.savefig('view_image.png')
#
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')
#
# # 图像特征化
# imgGen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# '''
# # ZCA白化
# imgGen = ImageDataGenerator(zca_whitening=True)
# # 图像旋转
# imgGen = ImageDataGenerator(rotation_range=90)
# # 图像移动
# imgGen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
# # 图像剪切
# imgGen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# '''
# imgGen.fit(X_train)
# for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot(331 + i)
#         plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     plt.savefig('feature_standard.png')
#     break


# # PyTorch——回归模型
# import torch
# from torch.autograd import Variable
# # 训练数据
# x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#     def forward(self, x):
#         y_pred = self.sigmoid(self.linear(x))
#         return y_pred
# # model
# model = Model()
# criterion = torch.nn.BCELoss(size_average=False)  # 损失函数
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# # 训练：前向反馈，损失，后向传播，step
# # training loop
# for epoch in range(500):
#     # forward pass
#     y_pred = model(x_data)
#     # compute loss
#     loss = criterion(y_pred, y_data)
#     if epoch % 20 == 0:
#         print(epoch, loss.data)
#     # zero gradients
#     optimizer.zero_grad()
#     #perform backward pass
#     loss.backward()
#     # update weights
#     optimizer.step()
# # after training
# hour_var = Variable(torch.Tensor([[0.5]]))
# print("predict (after training)", 0.5, model.forward(hour_var).data[0][0])
# hour_var = Variable(torch.Tensor([[7.0]]))
# print("predict (after training)", 7.0, model.forward(hour_var).data[0][0])


# # PyTorch——世界人口的线性回归
# import pandas as pd
# import torch
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
#
# current_path = r'C:/Users/Lenovo/Desktop/data/data'
# listName = current_path + r'/World_population_estimates.txt'
# df = pd.read_csv(listName, header=None, sep='\t')
# years = torch.tensor(df.iloc[:, 0], dtype=torch.float32)
# df[1] = df.apply(lambda x: float(''.join(x[1].split(',')[:])), axis=1)
# populations = torch.tensor(df[1], dtype=torch.float32)
#
# x = torch.stack([years, torch.ones_like(years)], 1)
# y = populations
# # 最小二乘法线性回归
# wr, _ = torch.gels(y, x)
# slope, intercept = wr[:2, 0]
# result = 'population = {:.2e}*year + {:.2e}'.format(slope, intercept)
# print('result: ' + result)
#
# # 正规方程法做线性回归
# w = x.t().mm(x).inverse().mm(x.t()).mv(y)
# slope, intercept = w
# result = 'population = {:.2e}*year+{:.2e}'.format(slope, intercept)
# print('result:' + result)
#
# #绘制线性回归的结果
# plt.xlabel('Year')
# plt.ylabel('Population')
# # 绘制散点
# plt.scatter(years,
#             populations,
#             s=0.1,
#             label='actual',
#             color='k')
# # 预测
# estimates = [slope*year + intercept for year in years]
# #绘制直线
# plt.plot(years.numpy(),
#          estimates,
#          label=result,
#          color='k')
# plt.legend()
# plt.savefig(current_path+'result.png')


