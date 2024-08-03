"""用于做市商订单的行为分析
1.基本假设：撤单是看不到的，但每个trades成交单都是对应1个参与方（机构交易者/量化参与者/套利者参与比例/散户/mm）
2.用模型划分这些参与方
3.并与数据系统里的进行对应

建模分析步骤：先从trades开始拆分试试，最近4小时内的trades就会有相应的数据
最大拉取1000条trades，从看看这一次的trades里面能分析出几个参与方，会不会有那种着急想买然后吃单的人，或者其他的散户。做个plot图出来
"""

# todo:1.反向研究，如实时获取订单薄的数据变化情况，分析其中哪些是做市商的单，哪些是散户的单，哪些是做市商主动吃掉了散户的单然后完成了更高级别套利的，哪些是做市商撤掉的单

import pandas as pd

df = pd.read_csv("trades.csv")
# df = df[['price', 'qty', 'isBuyerMaker', 'time']]
# maker成交单的数量和价格特点
df['quoteQty_1'] = round(df['quoteQty'], 2)
df['quoteQty_2'] = round(df['quoteQty'], 0)
df['qty_1'] = round(df['qty'], 1)
df['qty_2'] = round(df['qty'], 0)
# df = df[['id', 'price', 'qty', 'qty_1', 'qty_2', 'quoteQty', 'quoteQty_1', 'quoteQty_2', 'isBuyerMaker', 'time']]
# df.to_csv("trades.csv")


# taker成交单的数量和价格特点，下单时间等等
# 简单统计做市商的情况下什么类型赚的做多
num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 22, 23]
df['isBuyerMaker'] = df['isBuyerMaker'].apply(lambda x: -1 if x == False else 1)


def check_profit(df, num):
    res = df.groupby(['qty'])['qty'].count()
    order_qty_list = list(res[res > num].index)
    mm_df = df[df['qty'].isin(order_qty_list)]
    res = mm_df[['price', 'qty', 'quoteQty', 'isBuyerMaker']]
    profit = sum(res['quoteQty'] * res['isBuyerMaker'])  # 数值为正，表示买到了多少，而数值为负，表示卖出赚到的钱
    return profit, mm_df


mm_profit = {}
# for num in num_list:
#    mm_profit[num + 1] = check_profit(df, num)[0]

# print("做市商获利情况与频次潜在关系:%s" % mm_profit)
""" 
做市商获利情况与频次潜在关系:{2: 12077.933249999993, 3: 1701.9155700000201, 4: -5303.23208999999, 5: -1790.674090000004, 
6: -2804.8759099999993, 7: -3094.8324500000003, 8: -3084.8375600000013, 9: -1994.1890000000017, 10: 1705.0818300000014, 
11: 3069.99231, 13: 3089.96611, 14: 4339.394430000001, 18: 489.39442999999994, 23: 489.9221599999999, 24: 0}
"""


def mm_plot(df):
    import matplotlib.pyplot as plt
    # 创建画布和子图
    fig, ax1 = plt.subplots()

    # 绘制价格折线图
    ax1.plot(df.index, df['price'], color='blue', marker='.', label='Price')

    # 根据买卖情况在对应位置上标记符号
    buy_signals = df[df['isBuyerMaker'] == 1]
    sell_signals = df[df['isBuyerMaker'] == -1]
    ax1.scatter(buy_signals.index, buy_signals['price'], color='green', marker='o', label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['price'], color='red', marker='x', label='Sell Signal')

    # 添加买卖数量和价值的双y轴
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['qty'], color='orange', linestyle='--', label='Quantity')
    ax2.plot(df.index, df['quoteQty'], color='purple', linestyle='--', label='Quote Quantity')

    # 设置图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 添加标题和标签
    ax1.set_title('Price and Buy/Sell Signals over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Quantity / Quote Quantity')

    # 显示图形
    plt.show()


def buyers_analyze(df):
    # 散户分析
    df = df[df['isBuyerMaker'] == -1]
    import matplotlib.pyplot as plt
    # 创建画布和子图
    fig, ax1 = plt.subplots()

    # 绘制价格折线图
    ax1.plot(df['time'], df['price'], color='blue', marker='.', label='Price')

    # 根据买卖情况在对应位置上标记符号

    # 添加买卖数量和价值的双y轴
    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['qty'], color='orange', linestyle='--', label='Quantity')


    # 设置图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 添加标题和标签
    ax1.set_title('Price and Buy/Sell Signals over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Quantity / Quote Quantity')

    # 显示图形
    plt.show()


# mm_plot(check_profit(df[700:1000], 3)[1])
buyers_analyze(df)