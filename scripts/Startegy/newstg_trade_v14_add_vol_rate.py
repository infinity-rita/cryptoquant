"""
测试：修改后的策略12.29.2022
回测部分 -- 永续合约，1x只做空
加入成交情况判断程序

修改平仓条件，用ma25/ma7的条件
"""
import time

import numpy as np
import pandas as pd

from scripts.hf_predict_janestreet import run


class OptionVolatility(object):
    def __init__(self, data, type_, freq, data_type):
        self.close = data['Close']  # 四个指标都要用到，改为 self.data=data
        self.high = data['High']
        self.low = data['Low']
        self.time = data['TimeStamp']
        self.type = type_
        self.data = data[::3]  # 15min
        returns = self.Volatility
        self.freq = freq
        self.data_type = data_type
        self.basic_result = pd.DataFrame(
            {'tradingtime': self.time, 'close': self.close, 'low': self.low, 'high': self.high,
             'volatility': returns})  # 保持5min的结果
        self.basic_result['volatility'] = self.basic_result['volatility'].fillna(0)
        self.freqAdjust()  # 调整频率
        self.ma7 = self.close.rolling(7).mean()
        self.ma25 = self.close.rolling(25).mean()
        self.data_type = data_type
        self.avgprice = round((self.high + self.low) / 2, 2)
        returns = self.Volatility
        vol_rate = (self.high - self.low) / self.low  # 波动率
        base_return = (self.close / self.close.shift(1)) - 1
        same_percentage = round(base_return / vol_rate, 4)
        self.result = pd.DataFrame({'tradingtime': self.time, 'close': self.close, 'low': self.low, 'high': self.high,
                                    'avgprice': self.avgprice, 'volatility': returns, 'vol_rate': vol_rate,
                                    "same_percentage": same_percentage})  # 最终要输出的结果
        self.result['volatility'] = self.result['volatility'].fillna(0)
        self.result["same_percentage"] = self.result['same_percentage'].fillna(0)
        self.result.loc[self.result['avgprice'].isnull(), 'avgprice'] = round((self.result.loc[self.result[
                                                                                                   'avgprice'].isnull(), 'low'] +
                                                                               self.result.loc[self.result[
                                                                                                   'avgprice'].isnull(), 'high']) / 2,
                                                                              2)
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
        for i in range(998, len(self.result) - 3):  # 改成while,不断获取数据
            vol_list = self.result.iloc[i:i + 3, :]
            # 判断是啥信号，挂单撮合阶段
            pnl = 0.0  # 每单损益初始置零，每单收益率初始置零
            roi = 0.0
            if position == 0 and cash != 0:
                if self.StrongUpSignal_V3(vol_list):
                    self.signal_record.append('SU')
                    # 挂单未成交，position不改变
                    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    if alltradeqty != 0:
                        position = 1
                        tradeprice = self.ask_price
                    # 不再考虑弱上涨趋势，弱上涨还有可能是，上涨越来越乏力
                elif self.WeakUpSignal_V3(vol_list):
                    self.signal_record.append("WU")
                    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    if alltradeqty != 0:
                        position = 1
                        tradeprice = self.ask_price
                elif self.StrongDownSignal(vol_list):
                    self.signal_record.append("SD")
                    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    if alltradeqty != 0:
                        position = -1
                        tradeprice = self.ask_price
                elif self.WeakDownSignal(vol_list):
                    self.signal_record.append("WD")
                    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    if alltradeqty != 0:
                        position = -1
                        tradeprice = self.ask_price
                else:
                    pass
            elif self.LongClosed_V3(vol_list, i) and position == 1:
                self.signal_record.append("LC")
                position = 0
                alltradeqty, cash, pnl = self.CloseActionCheck_V2(i, position, alltradeqty)
                tradeprice = self.ask_price
            elif self.ShortClosed_V3(vol_list, i) and position == -1:
                self.signal_record.append("SC")
                position = 0
                alltradeqty, cash, pnl = self.CloseActionCheck_V2(i, position, alltradeqty)
                tradeprice = self.ask_price
            else:
                pass
            # orderbook成交状态更新
            if cash != 0 and pnl != 0:
                roi = pnl / (cash - pnl) * 100
                tot_roi = (cash - tot_value) / tot_value * 100
            basic_return = (self.result['close'].iloc[i + 2] - self.result['close'].iloc[0]) / \
                           self.result['close'].iloc[0] * 100
            pos_value = (tot_roi / 100 + 1) * tot_value
            orderbook = {'portfolio': 'ETH ', 'position': position, 'tradeprice': tradeprice,
                         'alltradeqty': alltradeqty, 'askprice': self.ask_price,
                         'cash': cash, 'pnl': pnl, 'roi(%)': roi, 'tot_roi(%)': tot_roi,
                         'portfolio_price': self.result['close'].iloc[i + 2], 'basic_return(%)': basic_return,
                         'pos_value': pos_value, "same_percentage": self.result['same_percentage'].iloc[i + 2]}
            current_time = self.timetodate(self.result['tradingtime'].iloc[i + 2])
            if len(self.trading_result):
                res = pd.DataFrame(orderbook, index=[current_time])
                self.trading_result = pd.concat([self.trading_result, res])
            else:
                self.trading_result = pd.DataFrame(orderbook, index=[current_time])
            # 实时存储结果
            self.trading_result.to_csv("rl_trading_result.csv")
        print("=============" + self.data_type + "_" + self.freq + "策略运行结束，交易结果如下所示=============")
        print(self.trading_result)

    def stoploss(self, alltradeqty, num):
        position = 0
        self.ask_price = self.result['close'].iloc[num + 2]
        pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'][-1]) * (
                self.trading_result['position'][-1] - position) - self.ask_price * alltradeqty * 0.04 / 100
        cash = pnl + self.trading_result['alltradeqty'][-1] * self.trading_result['tradeprice'][-1]
        if pnl / (cash - pnl) * 100 < -0.8:
            return True
        return False

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

    def StrongUpSignal_V4(self, vol_list):
        # 强上涨信号, 加入vol_rate的考虑
        # TODO:对比加入vol_rate对整体的影响
        returns = vol_list['volatility']
        same_per = vol_list['same_percentage']
        delta = 0  # 最低收益率
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'SC' and returns.iloc[1] >= delta and returns.iloc[2] > returns.iloc[1] and \
                    same_per.iloc[2] < same_per.iloc[1]:
                self.ask_price = vol_list['low'].iloc[2]
                # TODO：这里不用预测，因为此时的low价已经出现,直接进入check单的部分
                # print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
                return True
            elif returns.iloc[1] >= delta and returns.iloc[2] > returns.iloc[1] and same_per.iloc[2] < same_per.iloc[
                1]:  # 连续上涨，且上涨幅度越来越大
                self.ask_price = vol_list['close'].iloc[2]
                # TODO：理论上一定成交
                # print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
                return True
            else:
                pass
        elif returns.iloc[1] >= delta and returns.iloc[2] > returns.iloc[1] and same_per.iloc[2] < same_per.iloc[
            1]:  # 连续上涨，且上涨幅度越来越大
            self.ask_price = vol_list['close'].iloc[2]  # 以长期横盘价挂单(暂以最后时刻的均价挂单)
            # TODO：理论上一定成交
            # print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
            return True
        else:
            pass
        return False

    def StrongUpSignal_V3(self, vol_list):
        # 强上涨信号
        returns = vol_list['volatility']
        delta = 0  # 最低收益率
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'SC' and returns.iloc[1] >= delta and returns.iloc[2] > returns.iloc[1]:
                self.ask_price = vol_list['low'].iloc[2]
                # TODO：这里不用预测，因为此时的low价已经出现,直接进入check单的部分
                # print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
                return True
            elif returns.iloc[1] >= delta and returns.iloc[2] > returns.iloc[1]:  # 连续上涨，且上涨幅度越来越大
                self.ask_price = vol_list['close'].iloc[2]
                # TODO：理论上一定成交
                # print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
                return True
            else:
                pass
        elif returns.iloc[1] >= delta and returns.iloc[2] > returns.iloc[1]:  # 连续上涨，且上涨幅度越来越大
            self.ask_price = vol_list['close'].iloc[2]  # 以长期横盘价挂单(暂以最后时刻的均价挂单)
            # TODO：理论上一定成交
            # print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
            return True
        else:
            pass
        return False

    def StrongDownSignal(self, vol_list):
        # 强下跌信号
        delta = 0
        returns = vol_list['volatility']
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and returns.iloc[1] < -delta and returns.iloc[2] < returns.iloc[1]:
                self.ask_price = vol_list['close'].iloc[2]
                # TODO：理论上一定成交
                # print('======出现强下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
            elif returns.iloc[0] < -delta and returns.iloc[1] < returns.iloc[0] and returns.iloc[2] < returns.iloc[1]:
                self.ask_price = vol_list['close'].iloc[2]
                # TODO：理论上一定成交
                # print('======出现强下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
            else:
                pass
        elif returns.iloc[0] < -delta and returns.iloc[1] < returns.iloc[0] and returns.iloc[2] < returns.iloc[1]:
            self.ask_price = vol_list['close'].iloc[2]
            # TODO：理论上一定成交
            # print('======出现强下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
            return True
        else:
            pass
        return False

    def WeakUpSignal_V3(self, vol_list):
        # 弱上涨信号 修改为博反弹
        returns = vol_list['volatility']
        same_per = vol_list['same_percentage']
        delta = 0
        if len(self.signal_record) != 0:  # '原为LC'
            if self.signal_record[-1] == 'SC' and returns.iloc[1] >= delta and returns.iloc[2] > delta and \
                    same_per.iloc[2] < same_per.iloc[1]:  # 震荡行情，但不下跌
                self.ask_price = vol_list['close'].iloc[2]  # 取三个时段的平均价挂单做多(暂以最后时刻的均价挂单)
                # TODO：理论上一定成交
                # print('======出现弱上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
                return True
        return False

    def WeakDownSignal_v2(self, vol_list):
        # 弱下跌信号
        returns = vol_list['volatility']
        same_per = vol_list['same_percentage']
        delta = 0
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and returns.iloc[1] < -delta and returns.iloc[
                2] < -delta and same_per.iloc[2] > same_per.iloc[1]:  # 时间已经走到下一刻i+2了，这里用历史数据判断的意义是？
                self.ask_price = vol_list['close'].iloc[2]
                # TODO: 理论上一定成交
                # print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
            elif returns.iloc[0] < -delta and returns.iloc[1] < -delta and returns.iloc[2] < -delta and same_per.iloc[
                2] > same_per.iloc[1]:
                self.ask_price = vol_list['close'].iloc[2]  # TODO：理论上一定成交
                # print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
            else:
                pass
        elif returns.iloc[0] < -delta and returns.iloc[1] < -delta and returns.iloc[2] < -delta and same_per.iloc[2] > \
                same_per.iloc[1]:
            self.ask_price = vol_list['close'].iloc[2]  # TODO：理论上一定成交
            # print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
            return True
        else:
            pass
        return False

    def WeakDownSignal(self, vol_list):
        # 弱下跌信号
        returns = vol_list['volatility']
        delta = 0
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and returns.iloc[1] < -delta and returns.iloc[
                2] < -delta:  # 时间已经走到下一刻i+2了，这里用历史数据判断的意义是？
                self.ask_price = vol_list['close'].iloc[2]
                # TODO: 理论上一定成交
                # print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
            elif returns.iloc[0] < -delta and returns.iloc[1] < -delta and returns.iloc[2] < -delta:
                self.ask_price = vol_list['close'].iloc[2]  # TODO：理论上一定成交
                # print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
            else:
                pass
        elif returns.iloc[0] < -delta and returns.iloc[1] < -delta and returns.iloc[2] < -delta:
            self.ask_price = vol_list['close'].iloc[2]  # TODO：理论上一定成交
            # print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
            return True
        else:
            pass
        return False

    def ShortClosed(self, vol_list, num):
        # 平空
        delta = 0
        returns = vol_list['volatility']
        if self.signal_record[-1] in ['SD', 'WD']:
            if returns.iloc[2] > delta:  # 一旦下跌趋势中止就赶紧跑，不考虑什么波动率最大了
                # TODO:有一种方法是，根据历史数据预测returns.iloc[2]时刻的收益率，根据历史预测到，然后立刻用发送ask_price去，一般是可以成交的。
                # TODO:需x要预测一段时间，而不是一个点，比如预测接下来两个时间点的数据，然后就可以保证这一单一定成交
                # TODO:关键：预测必须低误差，足够准确，否则会错误做单
                self.ask_price = vol_list['close'].iloc[1]
                # self.ask_price = vol_list['close'].iloc[2]
                # print('======平空,平仓价格:$'+str(self.ask_price)+'======')
                return True
        return False

    def LongClosed(self, vol_list, num):
        # 平多
        delta = 0
        returns = vol_list['volatility']
        if self.signal_record[-1] in ['SU', 'WU']:
            if returns.iloc[2] <= -delta:
                self.ask_price = vol_list['close'].iloc[1]
                # self.ask_price = vol_list['close'].iloc[2]
                # TODO:情况同ShortClosed()
                # print('======平多,平仓价格:$'+str(self.ask_price)+'======')
                return True
        return False

    def ShortClosed_V3(self, vol_list, i):
        # 平空
        delta = 0
        data = self.data.iloc[i + 3 - 1000:i + 3]
        if self.signal_record[-1] in ['SD', 'WD']:
            if run(data) == 1:  # 一旦下跌趋势中止就赶紧跑，不考虑什么波动率最大了
                # TODO:有一种方法是，根据历史数据预测returns.iloc[2]时刻的收益率，根据历史预测到，然后立刻用发送ask_price去，一般是可以成交的。
                # TODO:需x要预测一段时间，而不是一个点，比如预测接下来两个时间点的数据，然后就可以保证这一单一定成交
                # TODO:关键：预测必须低误差，足够准确，否则会错误做单
                self.ask_price = vol_list['close'].iloc[1]
                # self.ask_price = vol_list['close'].iloc[2]
                # print('======平空,平仓价格:$'+str(self.ask_price)+'======')
                return True
        return False

    def LongClosed_V3(self, vol_list, i):
        # 平多
        # TODO:加入模型预测jane street
        delta = 0
        data = self.data.iloc[i + 3 - 1000:i + 3]
        if self.signal_record[-1] in ['SU', 'WU']:
            if run(data) == 0:
                self.ask_price = vol_list['close'].iloc[1]
                # self.ask_price = vol_list['close'].iloc[2]
                # TODO:情况同ShortClosed()
                # print('======平多,平仓价格:$'+str(self.ask_price)+'======')
                return True
        return False

    def ShortClosed_V2(self, vol_list, num):
        # 平空
        delta = 0
        same_per = vol_list['same_percentage']
        if self.signal_record[-1] in ['SD', 'WD']:
            if self.ma7.iloc[num + 2] > self.ma25.iloc[num + 2] and abs(
                    self.ma7.iloc[num + 2] - self.ma25.iloc[num + 2]) > abs(
                self.ma7.iloc[num + 1] - self.ma25.iloc[num + 1]):  # 一旦下跌趋势中止就赶紧跑，不考虑什么波动率最大了
                # TODO:有一种方法是，根据历史数据预测returns.iloc[2]时刻的收益率，根据历史预测到，然后立刻用发送ask_price去，一般是可以成交的。
                # TODO:需x要预测一段时间，而不是一个点，比如预测接下来两个时间点的数据，然后就可以保证这一单一定成交
                # TODO:关键：预测必须低误差，足够准确，否则会错误做单
                # self.ask_price = vol_list['low'].iloc[1]
                self.ask_price = vol_list['close'].iloc[2]
                # print('======平空,平仓价格:$'+str(self.ask_price)+'======')
                return True
        return False

    def LongClosed_V2(self, vol_list, num):
        # 平多
        delta = 0
        returns = vol_list['volatility']
        if self.signal_record[-1] in ['SU', 'WU']:
            if self.ma7.iloc[num + 2] < self.ma25.iloc[num + 2] and abs(
                    self.ma7.iloc[num + 2] - self.ma25.iloc[num + 2]) < abs(
                self.ma7.iloc[num + 1] - self.ma25.iloc[num + 1]):  # 一旦下跌趋势中止就赶紧跑，不考虑什么波动率最大了
                self.ask_price = vol_list['close'].iloc[2]
                # self.ask_price = vol_list['close'].iloc[2]
                # TODO:情况同ShortClosed()
                # print('======平多,平仓价格:$'+str(self.ask_price)+'======')
                return True
        return False

    def run(self, cash):
        return self.process(cash)

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


test_csv = ["2022-all-ETH-5m.csv"]
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
    # 测试1个月和1天的的数据：
    for k, v in secu_data.items():
        ST = OptionVolatility(v, type_='history', freq="15min", data_type=k)
        ST.run(cash=10000)
        res = ST.trading_result
        res.to_csv("trading_result.csv")
        print('最大回撤是：%s' % MaxDrawdown(res['pos_value']))
        print("单笔最大损失是：%s，单笔最大收益是：%s" % (min(res['roi(%)']), max(res['roi(%)'])))
        print("年化收益率：%s" % res['tot_roi(%)'].iloc[-1])

# signal_record只记录了上一次的持仓操作，但并不表示上一时刻是这个操作~, 检查判断是否有效
