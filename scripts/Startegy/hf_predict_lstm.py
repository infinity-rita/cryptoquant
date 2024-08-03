"""
"之前概率预测用的LSTM模型"
数据：根据时序做差分+归一化处理

改为监督模型，分类~
改DBN看看
"""
import pandas as pd
# -*- coding:UTF-8 -*-
import torch
from torch import nn



# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


def t_compare(x):
    if x > 0:
        x = 1
    if x < 0:
        x = -1


if __name__ == '__main__':
    # create database
    import numpy as np

    data = pd.read_csv('ETHUSDT-1m-2022-12.csv',
                       names=['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'xx', 'Amount', 'count', 'change1',
                              'chang2', 'chang3'])
    "只取前面100个数据来调整本文程序，并测试精度提升效果"
    data = data['Close']  # 改用1min的数据训练，15min波动过大，5min的波动相对也过大, 测试1min的结果
    # 单独预测一个值的结果，close的变化量是不是<0,然后提前操作一部分哈
    data_diff = data - data.shift(1)
    data_diff = data_diff[1:]
    dataset = data_diff
    # 调整为1分钟的数据
    # 用1年的数据滚动预测
    actual_return = []
    predict_return = []
    allyeardata = dataset
    # for i in range(len(allyeardata) - 100):
    for i in range(10):
        dataset = allyeardata[i * 14:100 + i * 14]  # 1天共1440分钟，分钟线数据1440条对应1天，最少以5天的数据进行测试？
        # 缩放dataset的结果
        # 重写需要这里的测试结果，
        # 待测试的结果：
        # dataset = (dataset - store_datamin) / (store_datamax - store_datamin)
        data_len = len(dataset)
        dataset = [-1 if x <= 0 else 1 for x in dataset]
        dataset = np.array(dataset).astype('float32')
        actual_return.append(dataset[-1])
        # print("实际三挡：%s" % dataset[-1])  # 不管预测的个数多少，只考虑最近的一个，put all in

        # choose dataset for training and testing
        test_15min_num = 14  # 预测15min的收益率是准的
        train_data_len = data_len - test_15min_num
        train_x = dataset[:train_data_len - 1]
        train_y = dataset[1:train_data_len]
        INPUT_FEATURES_NUM = 1
        OUTPUT_FEATURES_NUM = 1

        # test_x = train_x
        # test_y = train_y
        test_x = dataset[train_data_len - 1:-1]
        test_y = dataset[train_data_len:]

        # ----------------- train -------------------
        train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 5
        train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 5

        # transfer data to pytorch tensor
        train_x_tensor = torch.from_numpy(train_x_tensor)
        train_y_tensor = torch.from_numpy(train_y_tensor)
        # test_x_tensor = torch.from_numpy(test_x)
        "一层、16个隐藏神经元"
        import time

        time_start = time.time()
        lstm_model = LstmRNN(INPUT_FEATURES_NUM, 1, output_size=OUTPUT_FEATURES_NUM, num_layers=3)  # 16 hidden units
        # print('LSTM model:', lstm_model)
        # print('model.parameters:', lstm_model.parameters)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

        max_epochs = 200
        for epoch in range(max_epochs):
            output = lstm_model(train_x_tensor)
            loss = loss_function(output, train_y_tensor)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if loss.item() < 1e-4:
                # print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
                print("The loss value is reached")
                break
            elif (epoch + 1) % 100 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

        # prediction on training dataset
        predictive_y_for_training = lstm_model(train_x_tensor)
        predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

        # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files

        # ----------------- test -------------------
        # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
        lstm_model = lstm_model.eval()  # switch to testing model

        # prediction on test dataset
        test_x_tensor = test_x.reshape(-1, test_15min_num,
                                       INPUT_FEATURES_NUM)  # set batch size to 5, the same value with the training set
        test_x_tensor = torch.from_numpy(test_x_tensor)

        predictive_y_for_testing = lstm_model(test_x_tensor)
        predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
        time_end = time.time()
        # print('totally cost:', time_end - time_start, "s")
        # ------------------存数据 --------------
        # 获取数据predictive_y_for_testing
        # 预测结果
        # print("预测结果:%s" % predictive_y_for_testing[-1])
        predict_return.append(sum(predictive_y_for_testing))
        # data.to_csv('lstm-predict-test.csv')

    # 用1min的数据，计算一下滚动预测的情况
predict_return = [-1 if x <= 0 else -1 for x in predict_return]
res = 0
for i in range(len(predict_return)):
    if predict_return[i] == actual_return[i]:
        res += 1
print(res /len(predict_return))
