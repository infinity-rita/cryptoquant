"""
write by rita 1.21.2023
用于挖掘ma均线不同分钟频率，时间间隔的变化规律
-- should be done before 1.23

# 暂测试ETH 5m ma7 ma25

1.待测试ma99的情况，与ma7和ma25的关系
2.ma7和ma25差额值的情况
3.时间延迟条件
"""
import matplotlib.pyplot as plt  # 绘图
import pandas as pd

plt.show()
df = pd.read_csv('2022-all-ETH-5m.csv')

df['ma7'] = df['Close'].rolling(7).mean()

df['ma25'] = df['Close'].rolling(25).mean()

# TODO：构建金叉和死叉信号,仅计算第一次信号出现

df['up_signal'] = (df['ma7'] > df['ma25']).astype('int')

df['down_signal'] = (df['ma7'] < df['ma25']).astype('int')

df['cross_node'] = (df['ma7'] == df['ma25']).astype('int')  # 恰好形成交叉？因计算频率的不同，和close
# df['cross_node'] 求和为0

df.to_csv("datamining_ma.csv")

# 观察ma线和close价格的差异情况
test_df = df[['ma7', 'ma25', 'Close']]

# plt.plot(test_df[:100])

# ma7和ma25均线的差额
df['diff'] = df['ma25'] - df['ma7']  # 用来判别下跌信号，该指始终不为零，考虑15min带来的滞后情况
df['diff'] = df['diff'].fillna(1)
df['abs_diff'] = abs(df['diff'])
# TODO:用5min线计算ma7和ma25，做频率上的迁移，与用15min计算的ma7、ma25数据做对比，分析时间上的延迟

df_5min = df
df_15min = df[::3].reset_index(drop=True)
df_10min = df[::2].reset_index(drop=True)

# 15min线重新计算ma7/ma25
df_15min['ma7_15min'] = df_15min['Close'].rolling(7).mean()
df_15min['ma25_15min'] = df_15min['Close'].rolling(25).mean()
df_15min['diff_ma7_ma25'] = df_15min['ma7_15min'] - df_15min['ma25_15min']
df_15min['abs_diff_ma7_ma25'] = abs(df_15min['diff_ma7_ma25'])
df_15min['abs_diff_ma7_ma25'] = df_15min['abs_diff_ma7_ma25'].fillna(1)
df_15min['diff_ma7_5_15'] = abs(df_15min['ma7_15min'] - df_15min['ma7'])
df_15min['diff_ma25_5_15'] = abs(df_15min['ma25_15min'] - df_15min['ma25'])

df_15min.to_csv('ma_15min_5min_compare.csv')

diff_15min = df_15min[['diff_ma7_5_15', 'diff_ma25_5_15']]
diff_15min = diff_15min.fillna(1)
# plt.plot(diff_15min)

# 10min线重新计算ma7/ma25
df_10min['ma7_10min'] = df_10min['Close'].rolling(7).mean()
df_10min['ma25_10min'] = df_10min['Close'].rolling(25).mean()
df_10min['diff_ma7_ma25'] = df_10min['ma7_10min'] - df_10min['ma25_10min']
df_10min['abs_diff_ma7_ma25'] = abs(df_10min['diff_ma7_ma25'])
df_10min['abs_diff_ma7_ma25'] = df_10min['abs_diff_ma7_ma25'].fillna(10)
df_10min['diff_ma7_5_15'] = abs(df_10min['ma7_10min'] - df_10min['ma7'])
df_10min['diff_ma25_5_15'] = abs(df_10min['ma25_10min'] - df_10min['ma25'])
diff_10min = df_10min.fillna(1)

diff_10min.to_csv('ma_10min_5min_compare.csv')

diff_10min = diff_10min[['diff_ma7_5_15', 'diff_ma25_5_15']]

# 分析5min abs_diff的峰值和历史时段最高/最低价格的关系
# TODO：从这里test
his_highest = []
new_data = []
df_15min.loc[df_15min['abs_diff'] <= 1, 'abs_diff'] = 0  # 这里一定要用
import numpy as np

label = 0
his_price = 0
for i in range(1, len(df_15min)):
    new_data.append(df_15min['abs_diff'].iloc[i])
    if df_15min['abs_diff'].iloc[i] != 0:
        his_highest_price_index = np.argmax(new_data)
        if new_data[his_highest_price_index] > his_price:
            his_price = new_data[his_highest_price_index]
            if len(his_highest) != 0:
                his_highest.pop()
        his_index = his_highest_price_index + label
        if [his_highest_price_index + label, new_data[his_highest_price_index],
            df_15min['Close'].iloc[his_index]] in his_highest:
            his_highest.pop()
        his_highest.append(
            [his_highest_price_index + label, new_data[his_highest_price_index], df_15min['Close'].iloc[his_index]])
    else:
        new_data = []
        label = i + 1
        his_price = df_15min['abs_diff'].iloc[i + 1]

df_test = pd.DataFrame(his_highest, columns=['index', 'highest_diff', 'close_price'])
# TODO:
#  1.上述测试的结论，可以峰值点，作为开空/平空（多）的点位，峰值小的值忽略（阈值分析）
#  2.这里的阈值分析和开仓方向还需要考虑~
#    a.开仓方向：最开始打标签，先看下一时刻的价格，如果是下降，那么现在就是-1，做空的意思，并到下一时刻时平空，然后再看下下一个时刻，如果是上涨，此刻做多，Ok
#    ***********
#    打标签过程如下：

df_test['after_close_price'] = df_test['close_price'].shift(-1)

df_test.loc[df_test['after_close_price'] > df_test['close_price'], 'action'] = 1
df_test.loc[df_test['after_close_price'] < df_test['close_price'], 'action'] = -1
df_test.loc[df_test['after_close_price'] == df_test['close_price'], 'action'] = 0
# TODO:
#    ***********
#    计算理想收益率如下：start 10000u 1x  测试最终结果 1年收益率：893155795403154.5%

df_test['cash'] = 0
cash = 10000
cash_list = []
for i in range(len(df_test) - 1):
    cash += cash / df_test['close_price'].iloc[i] * df_test['action'].iloc[i] * (
            df_test['after_close_price'].iloc[i] - df_test['close_price'].iloc[i])
    cash_list.append(cash)
cash_list.append(cash)
df_test['cash'] = cash_list

df_test['ma7'] = [df_15min['ma7'].iloc[i] for i in df_test['index']]
df_test['ma25'] = [df_15min['ma25'].iloc[i] for i in df_test['index']]
df_test.loc[df_test['ma7'] > df_test['ma25'], 'ma_action'] = 1
df_test.loc[df_test['ma7'] < df_test['ma25'], 'ma_action'] = -1
df_test.loc[df_test['ma7'] == df_test['ma25'], 'ma_action'] = 0
df_test['compare'] = (df_test['action'] == df_test['ma_action']).astype('int64')

# 对比表明，传统ma比较方法只有35%-30%的准确率
# 用传统方法计算带来的收益率为: -89%
df_test['cash'] = 0
cash = 10000
cash_list = []
for i in range(len(df_test) - 1):
    cash += cash / df_test['close_price'].iloc[i] * df_test['ma_action'].iloc[i] * (
            df_test['after_close_price'].iloc[i] - df_test['close_price'].iloc[i])
    cash_list.append(cash)
cash_list.append(cash)
df_test['cash'] = cash_list
# 1.计算mean,count,std, 统计参数的计算与所取历史数据长度相关（注意时间延时带来的影响）

# TODO:
#    ***********
#    根据打好的标签，开始用jane street的模型做训练和预测
df_test['action'] = df_test['action'].fillna(0)
df_15min['action'] = 0
df_15min['action'] = [
    df_test['action'].iloc[df_test['index'].values == i].values[0] if i in df_test['index'].values else 0 for i in
    range(len(df_15min))]

# 新的action参数加入
df_test['compare_action_2'] = df_test['compare_action_2'].fillna(0)
df_15min['compare_action_2'] = 0
df_15min['compare_action_2'] = [
    df_test['compare_action_2'].iloc[df_test['index'].values == i].values[0] if i in df_test['index'].values else 0 for
    i in
    range(len(df_15min))]

df_test['compare_action_1'] = df_test['compare_action_1'].fillna(0)
df_15min['compare_action_1'] = 0
df_15min['compare_action_1'] = [
    df_test['compare_action_1'].iloc[df_test['index'].values == i].values[0] if i in df_test['index'].values else 0 for
    i in
    range(len(df_15min))]
import math

test_df_15min = df_15min['abs_diff'][:100]
mean_15min = test_df_15min.mean()
count_15min = len(test_df_15min)
std_15min = test_df_15min.std()
ci95_hi_15min = mean_15min + 1.96 * std_15min / math.sqrt(count_15min)
ci95_lo_15min = mean_15min - 1.96 * std_15min / math.sqrt(count_15min)

# 结果：ci95_hi_5min
# Out[13]: 6.3183172750930225
# ci95_lo_5min
# Out[14]: 3.941511296335411

# 结果：ci95_hi_15min
# Out[13]:6.894127495312286
# ci95_lo_15min
# Out[14]:4.8013970761163005

# 2.忽略峰值<ci95_lo_15min的部分,保留筛选后的可平/开仓点位
adj_his_highest = [[k, v] for [k, v] in his_highest if v > ci95_lo_15min]

# 3.给筛选后的峰值加标签，开多/空/平仓
# TODO:
#  1.假设最开始没有仓位, 用df['diff']或未调整前的ma7与ma25差异值
#  2.加入ma线越过k线
# df['diff'] = df['ma25'] - df['ma7'] 注意这个被减数可能不一样

action_dict = {"开空/平多点位": [[k, v] for [k, v] in adj_his_highest if
                           (df_15min['ma7'].iloc[k] > df_15min['ma25'].iloc[k]) and df_15min['Close'].iloc[k] >
                           df_15min['ma7'].iloc[k]],
               "开多/平空点位": [[k, v] for [k, v] in adj_his_highest if
                           (df_15min['ma7'].iloc[k] < df_15min['ma25'].iloc[k]) and df_15min['Close'].iloc[k] <
                           df_15min['ma7'].iloc[k]]}
# 一些有趣的数据探索~
# TODO:1.改变action_dict的方向，在最高点转换方向，和最准确的相似度达到了0.96, 1年收益率是：36630.97% (366倍)
df_test['compare_action_1'] = 0
df_test.loc[df_test['ma7'] > df_test['ma25'], 'compare_action_1'] = -1
df_test.loc[df_test['ma7'] < df_test['ma25'], 'compare_action_1'] = 1

# TODO:2.加入ma值越过k线的考虑 -- 准确率只有0.32啦~ 这样跑的回报率一年下来是：3320.75% (33倍)
diff_test = df_test[df_test['action'] != df_test['compare_action_1']].reset_index(drop=True)
diff_test = diff_test[
    ['index', 'highest_diff', 'close_price', 'after_close_price', 'action', 'ma7', 'ma25', 'compare_action_1']]

df_test['compare_action_2'] = 0
df_test.loc[((df_test['ma7'] > df_test['ma25']) & (df_test['close_price'] > df_test['ma7'])), 'compare_action_2'] = -1
df_test.loc[((df_test['ma7'] < df_test['ma25']) & (df_test['close_price'] < df_test['ma7'])), 'compare_action_2'] = 1
# 复原到1年的order情况~


# TODO:3.加入一些最小反弹的考虑

# 15mim 条件成立
# TODO:1.二次检验是否全年都满足这个图
#      2.fail

# TODO:2.每个点位有两种操作，不一定是在该时刻同时进行两个操作，如先平多，然后开空；有可能只做一个方向的操作，并且需要等待下一个信号到来，才能做下一个操作
# 这一部分需要好好想想，或许可以和阈值结合起来使用~


# TODO:1.模拟实盘计算最大diff的结果，与np.argmax计算的结果做对比，只比较相对index
arg_his_highest_index = np.array(his_highest)[:, :1]
adj_his_highest_1 = pd.DataFrame(his_highest, columns=['index', 'abs_diff', 'close'])
adj_his_highest_1 = adj_his_highest_1.drop_duplicates().reset_index(drop=True)
# TODO:1.从adj_his_highest_1里的abs_diff，找升高的，找从小到最大的回弹max
com_his_highest_index = np.unique(np.array(his_highest)[:, :1])
adj_his_highest = df_15min.iloc[com_his_highest_index]
adj_his_highest = adj_his_highest[['Close', 'ma7', 'ma25']]
# 实盘滚动获取数据，保证获取的不是未来数据
his_highest = []
new_data = []

label = 0
his_price = 0
for i in range(1, len(df_15min)):
    new_data.append(df_15min['abs_diff'].iloc[i])
    if df_15min['abs_diff'].iloc[i] != 0:
        his_highest_price_index = np.argmax(new_data)
        if new_data[his_highest_price_index] > his_price:
            his_price = new_data[his_highest_price_index]
            if len(his_highest) != 0:
                his_highest.pop()
        his_index = his_highest_price_index + label
        his_highest.append(
            [his_highest_price_index + label, new_data[his_highest_price_index], df_15min['Close'].iloc[his_index]])
    else:
        new_data = []
        label = i + 1
        his_price = df_15min['abs_diff'].iloc[i + 1]

# 4251/3034=1.4 多出了一半的无效数据，查验一下这部分对结果的影响  116729.98% 1167倍？
# 可去掉149个~

adj_his_highest_old = pd.DataFrame(his_highest, columns=['index', 'abs_diff', 'close'])
j = 0
max_recv = []
wait_interval = []
for i in range(len(adj_his_highest_1)):
    if adj_his_highest_1['index'].iloc[i] == adj_his_highest_old['index'].iloc[j]:  # 索引相同
        j += 1
    else:  # 索引不同
        tmp = adj_his_highest_old['abs_diff'].iloc[j] - adj_his_highest_1['abs_diff'].iloc[i]
        wait_interval.append(adj_his_highest_old['index'].iloc[j] - adj_his_highest_1['index'].iloc[i])
        max_recv.append(tmp)

# TODO:1.平均等待多少的时间会出现更高的点：wait_interval
#       a.wait_interval: 最小值为2，最大值为86，假如取最小的2作为等待，当出现第一个max时，等待2个时刻，看下一个max是否出现，否则开始操作；
#           若下一个max出现，则再等待2~，直到不出现，且+2后的abs_diff不等于0，则操作 (fail)
# 对adj_his_highest_1进行处理，对index+2属于此df的做去重
adj_his_highest_interval = []
for i in range(len(adj_his_highest_1) - 1):
    if adj_his_highest_1['index'].iloc[i] + 2 != adj_his_highest_1['index'].iloc[i + 1]:
        adj_his_highest_interval.append(i)

adj_his_highest_1 = adj_his_highest_1.iloc[adj_his_highest_interval]
adj_his_highest_1 = adj_his_highest_1.reset_index(drop=True)
# 全部+2
adj_his_highest_1['index'] = adj_his_highest_1['index'] + 2
new_df=df_15min.iloc[adj_his_highest_1['index']]
# TODO:
#   2.平均diff的变化 max_recv，统计一下同期最小max_recv
#   3.记录反弹的波谷进行平仓？ 对小回弹取2的abs_diff
#   4.不全部滚仓，每次只做20%的仓位
