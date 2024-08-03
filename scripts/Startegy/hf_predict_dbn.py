"""
Created on Fri Jan 23 2023

@author: Administrator

全部深度学习模型收敛到0.5的准确率 ，判断为模型未生效
"""
import numpy as np
import pandas as pd
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from sklearn.neural_network import BernoulliRBM
from tensorflow.keras.optimizers import SGD


class DBN():
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            hidden_layer,
            learning_rate_rbm=0.001, # 0.0001
            batch_size_rbm=100,
            n_epochs_rbm=30,
            verbose_rbm=1,
            random_seed_rbm=1300,
            activation_function_nn="sigmoid",
            learning_rate_nn=1e-2, # 1e-2=0.01
            batch_size_nn=100,
            n_epochs_nn=100,
            verbose_nn=1,
            decay_rate=0):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.hidden_layer = hidden_layer
        self.learning_rate_rbm = learning_rate_rbm
        self.batch_size_rbm = batch_size_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.verbose_rbm = verbose_rbm
        self.random_seed = random_seed_rbm
        self.activation_function_nn = activation_function_nn
        self.learning_rate_nn = learning_rate_nn
        self.batch_size_nn = batch_size_nn
        self.n_epochs_nn = n_epochs_nn
        self.verbose_nn = verbose_nn
        self.decay_rate = decay_rate
        self.weight_rbm = []
        self.bias_rbm = []
        self.test_rms = 0
        self.result = []
        self.model = Sequential()

    def pretraining(self):
        input_layer = self.x_train
        for i in range(len(self.hidden_layer)):
            # print("DBN Layer {0} Pre-training".format(i + 1))
            rbm = BernoulliRBM(n_components=self.hidden_layer[i],
                               learning_rate=self.learning_rate_rbm,
                               batch_size=self.batch_size_rbm,
                               n_iter=self.n_epochs_rbm,
                               verbose=self.verbose_rbm,
                               random_state=self.verbose_rbm)
            rbm.fit(input_layer)
            # size of weight matrix is [input_layer, hidden_layer]
            self.weight_rbm.append(rbm.components_.T)
            self.bias_rbm.append(rbm.intercept_hidden_)
            input_layer = rbm.transform(input_layer)
        # print('Pre-training finish.')

    def finetuning(self):
        # print('Fine-tuning start.')

        for i in range(0, len(self.hidden_layer)):
            if i == 0:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function_nn,
                                     input_dim=self.x_train.shape[1]))
            elif i >= 1:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function_nn))
            else:
                pass
            layer = self.model.layers[i]
            layer.set_weights([self.weight_rbm[i], self.bias_rbm[i]])
        if (self.y_train.ndim == 1):
            self.model.add(Dense(1, activation=None, kernel_regularizer=regularizers.l2(0.01)))
        else:
            self.model.add(Dense(self.y_train.shape[1], activation=None))

        #sgd = SGD(learning_rate=self.learning_rate_nn, decay=self.decay_rate)
        self.model.compile(loss='mse',
                           optimizer='nadam',
                           )
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size_nn,
                       epochs=self.n_epochs_nn, verbose=self.verbose_nn)
        # print('Fine-tuning finish.')
        self.test_rms = self.model.evaluate(self.x_test, self.y_test)

        self.result = np.array(self.model.predict(self.x_test))
        return self.result

    def predict(self, series):
        return np.array(self.model.predict_proba(series, 1))


if __name__ == '__main__':
    # create database
    data = pd.read_csv('ETHUSDT-1m-2022-12.csv',
                       names=['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'xx', 'Amount', 'count', 'change1',
                              'chang2', 'chang3'])
    "只取前面100个数据来调整本文程序，并测试精度提升效果"
    data = data['Close']  # 改用1min的数据训练，15min波动过大，5min的波动相对也过大, 测试1min的结果
    # 单独预测一个值的结果，close的变化量是不是<0,然后提前操作一部分哈
    data_diff = data - data.shift(1)
    data_diff = data_diff[1:]  # 1min部分预测
    dataset = data_diff
    dataset = [-1 if x <= 0 else 1 for x in dataset]
    dataset = np.array(dataset).astype('float32')
    dataset = dataset.reshape(-1, 1)
    actual_return = []
    predict_return = []
    allyeardata = dataset
    for i in range(1000): # 最终收敛到0.5
        dataset = allyeardata[i * 15:3000 + i * 15] # 1天1440条的数据，test 3天的用来学习
        actual_return.append(dataset[-1])
        data_len = len(dataset)
        # choose dataset for training and testing
        test_15min_num = 15  # 预测15min的收益率是准的
        train_data_len = data_len - test_15min_num
        train_x = dataset[:train_data_len - 1]
        train_y = dataset[1:train_data_len]
        test_x = dataset[train_data_len - 1:-1]
        test_y = dataset[train_data_len:]
        hidden_layer = [4]
        import time

        time_start = time.time()

        model = DBN(train_x, train_y, test_x, test_y, hidden_layer)
        model.pretraining()
        predict_result = model.finetuning()
        predict_return.append(sum(predict_result))
        time_end = time.time()
        # print('totally cost:', time_end - time_start, "s")

predict_return = [-1 if x <= 0 else -1 for x in predict_return]
res = 0
for i in range(len(predict_return)):
    if predict_return[i] == actual_return[i]:
        res += 1
print(res / len(predict_return))
