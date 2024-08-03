# 期权交易策略，隐含波动率和实际波动率

# 这个可以用来判断上涨或下跌的方向，在左侧交易法则中，用于预判方向

# 波动率策略（黑天鹅策略）
# 1.测试品种：altcoin/期权
"""
1.期权的波动率高，加上期权费会带来较大的回撤；波动率低时，买入看跌期权最为合适
2.altcoin的波动可参考资金费率，资金费率的上升，意味着做空机会
3.


参考文章：
https://zhuanlan.zhihu.com/p/43111892
https://www.zhihu.com/question/52172444
https://blog.csdn.net/m0_46665608/article/details/108413201

需要的数据：价格（5min的高频数据） 数据可获得性？需要近1年/3年。测试BTC/BTC/bnb的波动，先从股票开始

1.计算RV的期望（即实际波动）
2.高低价波动
3.GARCH波动
4.进一步分解为不对称波动
***********************TODO***********************************
"测试：获取近一个月来BTC 5min的数据，预测未来半小时内的波动率情况"

"波动率策略：多空"
"首先验证：1.上涨时的波动率大于下跌时的波动率"

Talib python包：https://zhuanlan.zhihu.com/p/88115372
"""
import datetime
import time

import numpy as np
import pandas as pd
#from binance.spot import Spot

#from tools.robot import send_msg


class OptionVolatility(object):
    def __init__(self, data, type_, freq, data_type):
        self.close = data['Close']
        self.type = type_
        self.freq = freq
        self.data_type = data_type

    def process(self):
        "根据历史波动率、预测波动率，按步长循环跑GARCH，一直获得预测值，则需要窗口函数"
        "以5min线数据为例，最近4小时内的波动率小于上一个4小时的波动率，且最近1天的平均波动率大于最近4小时的波动率,则为做空/平仓信号,-1"
        "反之，最近4小时内的波动率大于上一个4小时的波动率，或最近1天的平均波动率小于最近4小时的波动率，则为做多/平仓信号,1"
        # 先计算历史的，看历史的信号、收益率如何
        returns = self.Volatility
        vol_20x_min = returns.rolling(20).mean()  # 最近20小时的平均波动率 -- 5min线对应是100min
        vol_4x_min = returns.rolling(4).mean()  # 最近4小时的平均波动率 -- 5min线对应是20min
        vol_100x_min = returns.rolling(100).mean()  # 5min线对应是500min 8个半小时线
        res = []
        for i in range(1, len(vol_20x_min)):
            if vol_20x_min[i] > vol_4x_min[i] and vol_4x_min[i - 1] > vol_4x_min[i] and \
                    vol_100x_min[i] > vol_20x_min[i] and vol_20x_min[i - 1] > vol_20x_min[i]:
                res.append(-1)
            else:
                res.append(1)
        self.res = res
        return res

    def long(self):
        "多单持仓情况"
        holding_cost = 0  # 多单持有的BTC数量

    def short(self):
        "空单持仓情况"
        holding_cost = 0  # 空单持有的BTC数量
        pass

    def run(self, cash):
        "cash表示本金"
        start = cash
        self.close = self.freqAdjust()  # 调整频率
        res = self.process()
        total_value = [cash]
        # 测试历史情况，假设单向持仓，滚动持仓； 改成多空两个仓位的情况
        #time.sleep(5)
        #self.qqbot("=============" + self.data_type + "_" + self.freq + "策略开始运行=============")
        long_holding_cost = 0
        short_holding_cost = 0
        long_num = 0  # 标的的持仓数量，以BTC为例，多少枚BTC
        short_num = 0
        portfolio_value = 0  # 最开始都是0
        for i in range(100, len(res)):  # 根据最大rolling的windows大小
            num_ = (cash) / (self.close.iloc[i] * (1 + 0.04 / 100))  # 1对应全仓梭哈，100对应分配为100份进行策略
            tmp = self.close.iloc[i] * res[i - 1] * num_  # 手续费取最高的0.04%, 本单开仓的成本
            if res[i - 1] == 1 and sum(res[i - 5:i]) == 5 and long_num == 0:  # 连续5个是做多信号，且没有做多的仓位，则多
                if (cash - tmp * (1 + 0.04 / 100) > 0 or cash - tmp * (1 + 0.04 / 100) == 0):
                    #current_time = datetime.datetime.now()
                    #time.sleep(5)
                    #self.qqbot(str(current_time) + " 做多，开单价格：$" + str(self.close.iloc[i]))
                    long_holding_cost += tmp * (1 + 0.04 / 100)
                    long_num = num_
                    cash -= tmp * (1 + 0.04 / 100)
                    # portfolio_value = (long_num - short_num) * self.close.iloc[i]
                    portfolio_value = long_num * self.close.iloc[i]
                    short_num = 0
                    print(
                        "{:.0f},BTC价格：{:.2f},策略做多建仓，持仓成本：{:.2f}, 扣除手续费：{:.4f}, 账户现有持仓价值：{:.2f}, 账户剩余可用资金：{:.2f} u".format(
                            i, self.close.iloc[i], tmp, tmp * 0.04 / 100,
                            portfolio_value,
                            cash))
                    print(
                        "累积收益:{:.2f}, 累积持有BTC数量:{:.2f}".format(
                            portfolio_value + cash - start,
                            long_num))
            elif res[i - 1] == -1 and sum(res[i - 5:i]) == -5 and short_num == 0:  # 做空
                if (cash + tmp * (1 + 0.04 / 100) > 0 or cash + tmp * (1 + 0.04 / 100) == 0) or (
                        cash - tmp * (1 - 0.04 / 100) > 0 or cash - tmp * (
                        1 - 0.04 / 100) == 0 and portfolio_value + tmp > 0):  # res[i]=-1,做空或平仓
                    #current_time = datetime.datetime.now()
                    #time.sleep(5)
                    #self.qqbot(str(current_time) + " 做空，开单价格：$" + str(self.close.iloc[i]))
                    short_holding_cost += tmp * (1 + 0.04 / 100)  # 负数，多单平仓就+
                    short_num = cash / (self.close.iloc[i] * (1 + 0.04 / 100))
                    cash = 0
                    portfolio_value = short_num * self.close.iloc[i]
                    print(
                        "{:.0f},BTC价格：{:.2f},策略做空建仓，持仓成本：{:.2f}, 扣除手续费：{:.4f}, 账户现有持仓价值：{:.2f}, 账户剩余可用资金：{:.2f} u".format(
                            i, self.close.iloc[i], self.close.iloc[i] * res[i - 1] * short_num,
                                                   - self.close.iloc[i] * res[i - 1] * short_num * 0.04 / 100,
                            portfolio_value,
                            cash))
                    print(
                        "累积收益:{:.2f}, 累积持有BTC数量:{:.2f}".format(
                            portfolio_value + cash - start,
                            long_num - short_num))
            elif sum(res[i - 4:i]) == -4 and long_num != 0:  # 出现2个做空信号，则直接平多
                #current_time = datetime.datetime.now()
                #time.sleep(5)
                #self.qqbot(str(current_time) + " 平多，平仓价格：$" + str(self.close.iloc[i]) + " 平仓收益:" + str(
                #    round(long_num * self.close.iloc[i] * (1 - 0.04 / 100) - portfolio_value, 2)))
                cash += long_num * self.close.iloc[i] * (1 - 0.04 / 100)  # 平仓增加本金,并扣去手续费
                long_num = 0
            elif sum(res[i - 4:i]) == 4 and short_num != 0:  # 出现2个做多信号，则直接平空
                #current_time = datetime.datetime.now()
                #time.sleep(5)
                pnl = - short_num * self.close.iloc[i] * (1 - 0.04 / 100) + portfolio_value
                #self.qqbot(str(current_time) + " 平空，平仓价格：$" + str(self.close.iloc[i]) + " 平仓收益:" + str(
                #    round(pnl, 2)))
                cash += portfolio_value + pnl  # 平仓增加本金,并扣去手续费
                short_num = 0
            else:
                pass
            total_value.append(portfolio_value + cash)
        if short_num:  # 平多，转空单持仓，然后最后强行平空
            portfolio_value = - short_num * self.close.iloc[-1] + portfolio_value
            #time.sleep(5)
            #current_time = datetime.datetime.now()
            #self.qqbot(str(current_time) + "平空，平仓价格：$" + str(self.close.iloc[-1]) + " 平仓收益:" + str(
            #   round(portfolio_value + cash - total_value[-1], 2)))
        if long_num:  # 多单持仓，最后强行平多
            portfolio_value = long_num * self.close.iloc[-1]
            #time.sleep(5)
            #current_time = datetime.datetime.now()
            #self.qqbot(str(current_time) + "平多，平仓价格：$" + str(self.close.iloc[-1]) + " 平仓收益:" + str(
            #    round(portfolio_value + cash - total_value[-1], 2)))
        if not short_num and not long_num:  # 没有持仓
            portfolio_value = 0.0
        profit = (cash + portfolio_value - start) / start * 100
        profit_free = (self.close.iloc[-1] - self.close.iloc[0]) / self.close.iloc[0] * 100  # 实际波动率、收益率
        #time.sleep(5)
        #self.qqbot(self.data_type + "_" + self.freq + "测试时段内的收益率(百分数)：" + str(round(profit, 2)) + ", 标的收益率(百分数)：" + str(
        #    round(profit_free, 2)) + "")  # qq这个暂时不能识别 "%"，带这个字符会发不出
        print(self.data_type + "_" + self.freq + "测试时段内的收益率(百分数)：" + str(round(profit, 2)) + ", 标的收益率(百分数)：" + str(
            round(profit_free, 2)) + "")
        #time.sleep(5)
        #self.qqbot("=============" + self.data_type + "_" + self.freq + "策略运行结束=============")

    @property
    def Volatility(self):
        if self.type == "history":
            """ 计算回望型波动率-历史波动率 GARCH """
            self.preclose = self.close.shift(1)
            Daily_return = np.log(self.close / self.preclose)  # 计算daily return
            return Daily_return
        else:
            pass

    def freqAdjust(self):
        "修改数据的频率"
        if self.freq == "5min":
            pass
        elif self.freq == "10min":
            self.close = self.close[::2].reset_index(drop=True)
        elif self.freq == "15min":
            self.close = self.close[::3].reset_index(drop=True)
        elif self.freq == "30min":
            self.close = self.close[::6].reset_index(drop=True)
        elif self.freq == "60min":
            self.close = self.close[::12].reset_index(drop=True)
        else:
            pass
        return self.close

    def qqbot(self, info):
        resp_dict = {}
        resp_dict['msg_type'] = 'group'
        resp_dict['number'] = 648887836
        resp_dict['msg'] = info
        #send_msg(resp_dict)
        return


#client = Spot()
# print(client.klines("BTCUSDT", "5m"))
# 数据格式为：【timestamp,open,high,low,close...]
# secu_data = pd.DataFrame([x[1] for x in res['prices']], columns=['Close'])

secu_data_monthly = pd.read_csv("2021-all-BTC-5m.csv")
print(secu_data_monthly)
# secu_data = {"monthly": secu_data_monthly, "daily": secu_data_daily}
# secu_data = {"daily": secu_data_daily} 时间太短，暂时不用测
secu_data = {"monthly": secu_data_monthly}
# 测试1个月和1天的的数据：
for k, v in secu_data.items():
    ST = OptionVolatility(v, type_='history', freq="15min", data_type=k)
    ST.run(cash=10000)
