"""
加入ma均线之后测试策略的收益率

"""
import time

import numpy as np
import pandas as pd


class OptionVolatility(object):
    def __init__(self, data, type_, freq, data_type):
        self.close = data['Close']  # 四个指标都要用到，改为 self.data=data
        self.high = data['High']
        self.low = data['Low']
        self.time = data['TimeStamp']
        self.type = type_
        returns = self.Volatility
        self.freq = freq
        self.action = data['compare_action_1']
        self.data_type = data_type
        self.basic_result = pd.DataFrame(
            {'tradingtime': data['TimeStamp'], 'close': self.close, 'low': self.low, 'high': self.high,
             'volatility': returns})  # 保持5min的结果
        self.basic_result['volatility'] = self.basic_result['volatility'].fillna(0)
        self.avgprice = round((self.high + self.low) / 2, 2)
        returns = self.Volatility
        vol_rate = (self.high - self.low) / self.low  # 波动率
        base_return = (self.close / self.close.shift(1)) - 1
        same_percentage = round(base_return / vol_rate, 4)
        self.result = pd.DataFrame({'tradingtime': data['TimeStamp'], 'close': self.close, 'low': self.low, 'high': self.high,
                                    'avgprice': self.avgprice, 'volatility': returns, 'vol_rate': vol_rate,
                                    "same_percentage": same_percentage})  # 最终要输出的结果
        self.result.to_csv("result.csv")

    def timetodate(self, timestamp):
        timeArray = time.localtime(timestamp / 1000)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        return otherStyleTime

    def process(self, cash):
        self.signal_record = []  # 记录每一时刻的信号
        position = 0  # 最开始未开仓
        self.trading_result = {}
        self.ask_price = 0.0  # 最开始不竞价
        alltradeqty = 0.0
        tot_value = cash
        tot_roi = 0.0  # 总回报情况
        tradeprice = 0.0
        print("=============" + self.data_type + "_" + self.freq + "策略开始运行=============")
        for i in range(len(self.result)):  # 改成while,不断获取数据
            # 判断是啥信号，挂单撮合阶段
            pnl = 0.0  # 每单损益初始置零，每单收益率初始置零
            roi = 0.0
            if position == 0 and cash != 0:
                if self.action.iloc[i] == 1:  # 开多
                    self.signal_record.append('U')
                    # 挂单未成交，position不改变
                    self.ask_price = self.close.iloc[i]
                    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    if alltradeqty != 0:
                        position = 1
                        tradeprice = self.ask_price
                    # 不再考虑弱上涨趋势，弱上涨还有可能是，上涨越来越乏力
                elif self.action.iloc[i] == -1:  # 开空
                    self.signal_record.append("D")
                    self.ask_price = self.close.iloc[i]
                    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    if alltradeqty != 0:
                        position = -1
                        tradeprice = self.ask_price
                else:
                    pass
            elif self.action.iloc[i] == -1 and position == 1:  # 平多并立刻开空
                self.signal_record.append("LC")
                self.ask_price = self.close.iloc[i]
                position = 0
                alltradeqty, cash, pnl = self.CloseActionCheck_V2(i, position, alltradeqty)
                tradeprice = self.ask_price
                # 再立刻开空
                alltradeqty = cash * (1 - 0.04 / 100) / (self.ask_price)
                position = -1
                tradeprice = self.ask_price
            elif self.action.iloc[i] == -1 and position == -1:  # 平空并立刻开空，滚仓
                self.signal_record.append("LC")
                self.ask_price = self.close.iloc[i]
                position = -1
                alltradeqty, cash, pnl = self.CloseActionCheck_V2(i, position, alltradeqty)
                # 再立刻开空
                alltradeqty = cash * (1 - 0.04 / 100) / (self.ask_price)
                position = -1
                tradeprice = self.ask_price
            elif self.action.iloc[i] == 1 and position == 1:  # 平多并立刻开多
                self.signal_record.append("SC")
                self.ask_price = self.close.iloc[i]
                position = 1
                alltradeqty, cash, pnl = self.CloseActionCheck_V2(i, position, alltradeqty)
                # 再立刻开多
                alltradeqty = cash * (1 - 0.04 / 100) / (self.ask_price)
                position = 1
                tradeprice = self.ask_price
            elif self.action.iloc[i] == 1 and position == -1:  # 平空并立刻开多
                self.signal_record.append("SC")
                self.ask_price = self.close.iloc[i]
                position = 0
                alltradeqty, cash, pnl = self.CloseActionCheck_V2(i, position, alltradeqty)
                # 再立刻开多
                alltradeqty = cash * (1 - 0.04 / 100) / (self.ask_price)
                position = 1
                tradeprice = self.ask_price
            else:
                pass
            # orderbook成交状态更新
            if cash != 0 and pnl != 0:
                roi = pnl / (cash - pnl) * 100
                tot_roi = (cash - tot_value) / tot_value * 100
            basic_return = (self.result['close'].iloc[i] - self.result['close'].iloc[0]) / \
                           self.result['close'].iloc[0] * 100
            pos_value = (tot_roi / 100 + 1) * tot_value
            orderbook = {'portfolio': 'ETH ', 'position': position, 'tradeprice': tradeprice,
                         'alltradeqty': alltradeqty, 'askprice': self.ask_price,
                         'cash': cash, 'pnl': pnl, 'roi(%)': roi, 'tot_roi(%)': tot_roi,
                         'portfolio_price': self.result['close'].iloc[i], 'basic_return(%)': basic_return,
                         'pos_value': pos_value, "same_percentage": self.result['same_percentage'].iloc[i]}
            current_time = self.timetodate(self.result['tradingtime'].iloc[i])
            if len(self.trading_result):
                res = pd.DataFrame(orderbook, index=[current_time])
                self.trading_result = pd.concat([self.trading_result, res])
            else:
                self.trading_result = pd.DataFrame(orderbook, index=[current_time])
            # 实时存储结果
            self.trading_result.to_csv("rl_trading_result.csv")
            print(cash)
        print("=============" + self.data_type + "_" + self.freq + "策略运行结束，交易结果如下所示=============")
        print(self.trading_result)

    def bidActionCheck(self, num, cash):
        # 成交情况判断
        alltradeqty = 0.0
        pnl = 0.0
        if self.result['high'].iloc[num + 3] >= self.ask_price and self.result['low'].iloc[
            num + 3] <= self.ask_price:  # 挂单成交
            alltradeqty = cash * (1 - 0.02 / 100) / (self.ask_price)  # Maker按0.02%收手续费
            cash = 0
            # print("挂单成交!成交价格:$%s"%self.ask_price)
        return [alltradeqty, cash, pnl]

    def bidAction(self, num, cash):
        # 假设挂单全部成交
        pnl = 0.0
        alltradeqty = cash * (1 - 0.04 / 100) / (self.ask_price)
        cash = 0
        # print("挂单成交!成交价格:$%s"%self.ask_price)
        return [alltradeqty, cash, pnl]

    def CloseAction(self, num, position, alltradeqty):
        # 平仓操作，即刻以close价平仓，默认一定成交， pnl包括平仓手续费
        pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'].iloc[num - 1]) * (
                self.trading_result['position'].iloc[
                    num - 1] - position) - self.ask_price * alltradeqty * 0.04 / 100
        cash = pnl + self.trading_result['alltradeqty'].iloc[num - 1] * self.trading_result['tradeprice'].iloc[num - 1]
        return [alltradeqty, cash, pnl]

    def CloseActionCheck_V2(self, num, position, alltradeqty):
        # 平仓操作，加入成交情况判断
        # TODO:以下的判断都是在预测结果出现后的情况，重点修改这里
        # TODO:还需要判断强平的状态 5min的不用特别去写，和一般的情况是一样的
        # print("平仓成功!成交价格:$%s" % self.ask_price)
        pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'][-1]) * (
                self.trading_result['position'][-1] - position) - self.ask_price * alltradeqty * 0.04 / 100
        cash = pnl + self.trading_result['alltradeqty'][-1] * self.trading_result['tradeprice'][-1]
        return [alltradeqty, cash, pnl]

    def CloseActionCheck(self, num, position, alltradeqty):
        # 平仓操作，加入成交情况判断
        # TODO:以下的判断都是在预测结果出现后的情况，重点修改这里
        # TODO:还需要判断强平的状态 5min的不用特别去写，和一般的情况是一样的
        if self.result['high'].iloc[num + 2] >= self.ask_price and self.result['low'].iloc[
            num + 2] <= self.ask_price:  # 挂单成交
            # print("平仓成功!成交价格:$%s" % self.ask_price)
            pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'].iloc[num - 1]) * (
                    self.trading_result['position'].iloc[
                        num - 1] - position) - self.ask_price * alltradeqty * 0.02 / 100
            cash = pnl + self.trading_result['alltradeqty'].iloc[num - 1] * self.trading_result['tradeprice'].iloc[
                num - 1]
            return [alltradeqty, cash, pnl]
        else:
            if self.signal_record[-1] == 'SC':
                print('挂单平仓价未成交，以实时最低价吃单!')
                self.ask_price = self.basic_result['low'].iloc[(num + 2) * 3]
                # TODO:但是这个挂单也不能确定是否成交，需要预测
                # TODO:这里是对订单薄强行吃单
                pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'][-1]) * (
                        self.trading_result['position'][-1] - position) - self.ask_price * alltradeqty * 0.04 / 100
                cash = pnl + self.trading_result['alltradeqty'][-1] * self.trading_result['tradeprice'][-1]
                return [alltradeqty, cash, pnl]
            elif self.signal_record[-1] == 'LC':
                print('挂单平仓价未成交，以实时最高价吃单!')
                self.ask_price = self.basic_result['high'].iloc[(num + 2) * 3]
                # TODO:但是这个挂单也不能确定是否成交，需要预测
                # TODO:这里是对订单薄强行吃单
                pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'][-1]) * (
                        self.trading_result['position'][-1] - position) - self.ask_price * alltradeqty * 0.04 / 100
                cash = pnl + self.trading_result['alltradeqty'][-1] * self.trading_result['tradeprice'][-1]
                return [alltradeqty, cash, pnl]
            else:
                pass

    @property
    def Volatility(self):
        if self.type == "history":
            """ 计算回望型波动率-历史波动率 GARCH """
            p_return = np.log(self.close / self.close.shift(1))  # 计算给定频率时段内的return
            return p_return

    def freqAdjust(self):
        "修改数据的频率"
        if self.freq == "5min":
            pass
        elif self.freq == "10min":
            self.close = self.close[::2].reset_index(drop=True)
            self.high = self.high[::2].reset_index(drop=True)
            self.low = self.low[::2].reset_index(drop=True)
            self.time = self.time[::2].reset_index(drop=True)
        elif self.freq == "15min":
            self.close = self.close[::3].reset_index(drop=True)
            self.high = self.high[::3].reset_index(drop=True)
            self.low = self.low[::3].reset_index(drop=True)
            self.time = self.time[::3].reset_index(drop=True)
        elif self.freq == "30min":
            self.close = self.close[::6].reset_index(drop=True)
            self.high = self.high[::6].reset_index(drop=True)
            self.low = self.low[::6].reset_index(drop=True)
            self.time = self.time[::6].reset_index(drop=True)
        elif self.freq == "60min":
            self.close = self.close[::12].reset_index(drop=True)
            self.high = self.high[::12].reset_index(drop=True)
            self.low = self.low[::12].reset_index(drop=True)
            self.time = self.time[::12].reset_index(drop=True)
        else:
            pass


# 计算最大回撤：
def MaxDrawdown(return_list):
    # return_list:产品净值,即pos_value,返回%
    i = np.argmax((np.maximum.accumulate(return_list) - return_list))  # 最大回撤结束的位置
    if i == 0:
        j = 0
    else:
        j = np.argmax(return_list.iloc[:i])  # 回撤开始的位置
    return (return_list.iloc[j] - return_list.iloc[i]) / (return_list.iloc[j]) * 100


test_csv = ["add_action_15min_eth.csv"]
# test_csv=["2019-all-BTC-5m.csv","2020-all-BTC-5m.csv","2021-all-BTC-5m.csv","2022-yearly-newest-BTC-5m.csv"]
# test_csv = ["2019-all-ETH-5m.csv", "2020-all-ETH-5m.csv", "2021-all-ETH-5m.csv", "2022-all-ETH-5m.csv"]

# test_csv=["BTCUSDT-5m-2022-01.csv","BTCUSDT-5m-2022-02.csv","BTCUSDT-5m-2022-03.csv","BTCUSDT-5m-2022-04.csv",
#          "BTCUSDT-5m-2022-05.csv","BTCUSDT-5m-2022-06.csv","BTCUSDT-5m-2022-07.csv","BTCUSDT-5m-2022-08.csv",
#          "BTCUSDT-5m-2022-09.csv","BTCUSDT-5m-2022-10.csv","BTCUSDT-5m-2022-11.csv","2022-m12-newest-BTC-5m.csv"]
# test_csv=['2022-m12-newest-BTC-5m.csv']
for i in test_csv:
    # secu_data_monthly = pd.read_csv(i,header=None,names=['TimeStamp','Open','High','Low','Close','Volume','xx','Amount','count','change1','chang2','chang3'])
    secu_data_monthly = pd.read_csv(i)
    secu_data = {"monthly": secu_data_monthly}
    print(secu_data)
    # 测试1个月和1天的的数据：
    for k, v in secu_data.items():
        ST = OptionVolatility(v, type_='history', freq="15min", data_type=k)
        ST.process(cash=10000)
        res = ST.trading_result
        res.to_csv("trading_result.csv")
        print('最大回撤是：%s' % MaxDrawdown(res['pos_value']))
        print("单笔最大损失是：%s，单笔最大收益是：%s" % (min(res['roi(%)']), max(res['roi(%)'])))
        print("年化收益率：%s" % res['tot_roi(%)'].iloc[-1])

# signal_record只记录了上一次的持仓操作，但并不表示上一时刻是这个操作~, 检查判断是否有效
