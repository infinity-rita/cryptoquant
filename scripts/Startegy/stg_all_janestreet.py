"""
构造新的action，历史数据打标签
低买高卖，action取值0或1，确定开多或开空，or 平仓
"""
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
test_csv = "2022-m12-newest-BTC-5m.csv"
data = pd.read_csv(test_csv)
data['action'] = None
# 最开始position=0，表示没有仓位
position = 0
i = 0
while i < len(data):
    highest = data['Close'].iloc[i]
    lowest = data['Close'].iloc[i]
    close = data['Close'].iloc[i + 1]
    print("第%s次操作:" % (i + 1))
    if position == 0:
        list_high = []  # [1,2,3,4,2]
        if close > highest:
            list_high.append(close)
            data['action'].iloc[i + 1] = 1
            position = 1  # 开多
            print("开多")
        else:
            list_high = [close]
            data['action'].iloc[i + 1] = 1  # 平多/开空 print(list_high)
            position = -1  # 开空
            print("开空")
    if position == 1:  # 准备平多
        list_high = []  # [1,2,3,4,2]
        if close > highest:
            list_high.append(close)
        else:
            list_high = [close]
            data['action'].iloc[i+1] = 1  # 平多/开空 print(list_high)
            position = 0  # 平多
            print("平多")
    if position == -1:  # 准备平空
        list_high = []  # [1,2,3,4,2]
        if close < highest:
            list_high.append(close)
        else:
            list_high = [close]
            data['action'].iloc[i + 1] = 1  # 平空
            position = 0  #
            print("平空")
    print("action:%s，position:%s" % (data['action'].iloc[i + 1], position))
    i = i + 1
