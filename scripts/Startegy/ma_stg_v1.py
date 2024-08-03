"""
write by rita 1.27.2023
修改：
1.策略触发后，如果下一时刻方向未改变，则在交叉处平仓，不反向开仓;;;fail
2.尝试在max(abs(ma7-ma25)这一因子的基础上，加入ma7 max的判断
  1)只跟ma7线走，到高点做空，到低点做多  fail

先上实盘跑目前的哈，测试1月份的情况 ok
测试一下，用历史的ma导入进来
"""
import math

import matplotlib.pyplot as plt  # 绘图
import numpy as np
import pandas as pd

plt.show()


class MAStg(object):
    def __init__(self, data, freq, per):
        self.freq = freq
        self.data = data
        self.data = self.datahander(self.data)
        self.freqAdjust()
        self.per = per  # 仓位的百分比

    def datahander(self, df):
        df['ma7'] = df['Close'].rolling(7).mean()
        df['ma25'] = df['Close'].rolling(25).mean()
        df['diff'] = df['ma25'] - df['ma7']
        # 改为ma7和close的价差计算因子,信号出现的延时会更小
        # df['diff'] = df['Close'] - df['ma25']
        df['diff'] = df['diff'].fillna(1)
        df['abs_diff'] = abs(df['diff'])
        # 过滤掉前面用来计算ma的数据
        # df = self.dynamic_thershod(df)
        return df

    def dynamic_thershod(self, df):
        test_df_15min = df['abs_diff']  # 固定阈值anyway
        mean_15min = test_df_15min.mean()
        count_15min = len(test_df_15min)
        std_15min = test_df_15min.std()
        # ci95_hi_15min = mean_15min + 1.96 * std_15min / math.sqrt(count_15min)
        ci95_lo_15min = mean_15min - 1.96 * std_15min / math.sqrt(count_15min)
        df.loc[df['abs_diff'] <= ci95_lo_15min, 'abs_diff'] = 0  # 这里一定要用
        return df

    def process(self, df):
        his_highest = []
        new_data = []
        label = 0
        df.loc[df['abs_diff'] <= 3, 'abs_diff'] = 0  # 这里一定要用
        his_price = 0
        for i in range(1, len(df) - 1):
            new_data.append(df['abs_diff'].iloc[i])
            if df['abs_diff'].iloc[i] != 0:
                his_highest_price_index = np.argmax(new_data)
                if new_data[his_highest_price_index] > his_price:
                    his_price = new_data[his_highest_price_index]
                    if len(his_highest) != 0:
                        his_highest.pop()
                his_index = his_highest_price_index + label
                his_highest.append(
                    [his_highest_price_index + label, new_data[his_highest_price_index],
                     df['Close'].iloc[his_index]])
            else:
                new_data = []
                label = i + 1
                his_price = df['abs_diff'].iloc[i + 1]
        # +2去重+2
        # 对adj_his_highest_1进行处理，对index+2属于此df的做去重
        his_highest = pd.DataFrame(his_highest, columns=['index', 'abs_diff', 'close'])
        com_his_highest_index = np.unique(np.array(his_highest)[:, :1])
        return com_his_highest_index

    def cal(self):
        df = self.data
        com_his_highest_index = self.process(df)
        com_his_highest_index = np.array(list(com_his_highest_index)) + 1
        adj_df_test = df.iloc[com_his_highest_index]
        com_his_highest_index = list(com_his_highest_index)
        adj_df_test = adj_df_test[['Close', 'ma7', 'ma25', 'abs_diff']]
        adj_df_test['after_close_price'] = adj_df_test['Close'].shift(-1)
        adj_df_test['ma_action'] = 0
        adj_df_test.loc[adj_df_test['ma7'] > adj_df_test['ma25'], 'ma_action'] = -1
        adj_df_test.loc[adj_df_test['ma7'] < adj_df_test['ma25'], 'ma_action'] = 1
        # 加入后续检查的判断
        change_list = {}
        for i in range(len(adj_df_test) - 1):
            if adj_df_test['ma_action'].iloc[i] == adj_df_test['ma_action'].iloc[i + 1]:
                index_l = int(com_his_highest_index[i])
                index_h = int(com_his_highest_index[i + 1])
                for k in list(df['abs_diff'].iloc[index_l:index_h]):
                    if k == 0.0:
                        # 平仓并换方向
                        change_list[i] = int(k + 1)
        # adj_df_test.loc[
        #    (adj_df_test['ma7'] > adj_df_test['ma25']) & (adj_df_test['Close'] > adj_df_test['ma7']), 'ma_action'] = -1
        # adj_df_test.loc[
        #    (adj_df_test['ma7'] < adj_df_test['ma25']) & (adj_df_test['Close'] < adj_df_test['ma7']), 'ma_action'] = 1
        adj_df_test['cash'] = 0
        cash = 10000
        cash_list = [cash]
        for i in range(len(adj_df_test) - 1):
            pnl = cash * self.per / adj_df_test['Close'].iloc[i] * adj_df_test['ma_action'].iloc[i] * (
                    adj_df_test['after_close_price'].iloc[i] - adj_df_test['Close'].iloc[i])
            cash += pnl
            cash_list.append(cash)
        adj_df_test['cash'] = cash_list
        adj_df_test['roi(%)'] = (adj_df_test['cash'].shift(-1) / adj_df_test['cash'] - 1) * 100
        adj_df_test.to_csv('test.csv')
        return adj_df_test

    def freqAdjust(self):
        "修改数据的频率"
        if self.freq == "5min":
            pass
        elif self.freq == "10min":
            self.data = self.data[::2].reset_index(drop=True)
        elif self.freq == "15min":
            self.data = self.data[::3].reset_index(drop=True)
        elif self.freq == "30min":
            self.data = self.data[::6].reset_index(drop=True)
        elif self.freq == "60min":
            self.data = self.data[::12].reset_index(drop=True)
        else:
            pass

    # 计算最大回撤：
    def MaxDrawdown(self, return_list):
        # return_list:产品净值,即pos_value,返回%
        i = np.argmax((np.maximum.accumulate(return_list) - return_list))  # 最大回撤结束的位置
        if i == 0:
            j = 0
        else:
            j = np.argmax(return_list.iloc[:i])  # 回撤开始的位置
        return (return_list.iloc[j] - return_list.iloc[i]) / (return_list.iloc[j]) * 100

    def run(self):
        adj_df_test = self.cal()
        print('最大回撤是：%s' % self.MaxDrawdown(adj_df_test['cash']))
        print("单笔最大损失是：%s，单笔最大收益是：%s" % (min(adj_df_test['roi(%)']), max(adj_df_test['roi(%)'])))
        print("年化收益率：%s" % ((adj_df_test['cash'].iloc[-1] - 10000) / 10000 * 100))
        return adj_df_test
        # plt.plot(adj_df_test['cash'])


#test_csv = ["2023-01-ETH-5m.csv"]


# test_csv = ["2022-all-ETH-5m.csv"]
test_csv = ["2019-all-ETH-5m.csv", "2020-all-ETH-5m.csv", "2021-all-ETH-5m.csv", "2022-all-ETH-5m.csv"]


def read_his_ma():
    # 不用读历史，直接实盘接上即可
    pass


for i in test_csv:
    df = pd.read_csv(i)
    # 接上之前历史的数据，来读取最新的数值，
    print("******************")
    app = MAStg(df, freq='5min', per=1)
    res = app.run()
