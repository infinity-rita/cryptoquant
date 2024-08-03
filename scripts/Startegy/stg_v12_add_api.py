"""
# 1.10 TODO：1.优化策略的判断条件 2.私募 1.11 ok，10u调试接口(强平价格部分修改;强平信号到来前后修改)
# 根据V10的版本进行修改

1.11 进入测试阶段
"""
import time

import numpy as np
import pandas as pd
from binance.client import Client


def timetodate(timestamp):
    timeArray = time.localtime(timestamp / 1000)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


class RealtimeStgTest(object):
    def __init__(self):
        self.signal_record = []  # 记录每一时刻的信号
        self.trading_result = {}  # 策略运行结果记录
        self.ask_price = 0.0  # 最开始不竞价
        self.position = 0  # 最开始不持仓

    @property
    def connect(self):
        # 定时15min重连，确保和服务器保持连接
        api_key = "dxFzobejrXqyKYRjE3SB341m7w4oj18Y8c1hl9RRbibT0YCSPaOV6Y1JANnHsQPc"
        api_secret = "2BaxckpFPF4uLQ8psXkpukjQA8GPVXbgPvWnO9kDakRosqqLVoVC1oSMjjmGmp2j"
        bfa = Client(api_key=api_key, api_secret=api_secret)
        return bfa

    def get(self):
        "获取计算需要的数据"
        self.bfa = self.connect
        self.his_data = self.get_historical_kline_data(symbol="ETHUSDT", interval="15m",
                                                       start_time=round((time.time() - 15 * 60 * 4 - 1) * 1000))
        self.rt_data = self.get_latest_kline_data(symbol="ETHUSDT", interval="15m")
        self._list = [self.his_data['return'].iloc[-2], self.his_data['return'].iloc[-1],
                      np.log(self.rt_data['close'] / self.his_data['close'].iloc[-1])]  # 策略滚动测试部分

    def get_latest_kline_data(self, symbol, interval):
        '调用接口,通过api获取最新的数据'
        data = self.bfa.futures_continous_klines(pair=symbol, contractType="PERPETUAL", interval=interval, limit=1)[0][
               :8]  # 获取最新的15min klines数据
        data_list = {'StartTime': timetodate(data[0]), 'open': float(data[1]), 'high': float(data[2]),
                     'low': float(data[3]), 'close': float(data[4]), 'volume': float(data[5]),
                     'EndTime': timetodate(data[6]), 'amount': float(data[7])}
        return data_list

    def get_historical_kline_data(self, symbol, interval, start_time):
        '获取历史时刻kline数据,limit 3 计算临近2time return'
        # 香港ip连
        data = self.bfa.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_time,
                                                  limit=3)  # 获取历史的15min klines数据
        data = np.array(data)[:, :8]
        data_list = pd.DataFrame(data,
                                 columns=['StartTime', 'open', 'high', 'low', 'close', 'volume', 'EndTime', 'Amount'])
        data_list = data_list.astype('float64')
        data_list['StartTime'] = data_list['StartTime'].apply(timetodate)
        data_list['EndTime'] = data_list['EndTime'].apply(timetodate)
        data_list['return'] = np.log(data_list['close'] / data_list['close'].shift(1))
        data_list['return'] = data_list['return'].fillna(0)
        return data_list

    def place_maker_order(self, side):
        '做maker,建仓/平仓,limit下单,以最优价格成交'
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="LIMIT",
                                                             timeInForce="GTC",
                                                             quantity=round(self.tot_value / self.ask_price, 2),
                                                             price=self.ask_price, newOrderRespType="RESULT")

    @property
    def trade(self):
        try:
            self.bfa.futures_get_order(symbol="ETHUSDT", orderId=self.place_order_res['orderId'])
            self.bfa.futures_cancel_order(symbol="ETHUSDT", orderId=self.place_order_res[
                'orderId'])
            return False
        except:
            return True

    def liqudation(self, side):
        '做taker,强平/maker,止盈止损单'
        if side == 'BUY':
            self.ask_price = self.rt_data['low']
        if side == 'SELL':
            self.ask_price = self.rt_data['high']
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="TAKE_PROFIT_MARKET",
                                                             closePosition="TRUE", workingType='CONTRACT_PRICE',
                                                             priceProtect='TRUE',
                                                             stopPrice=self.ask_price, newOrderRespType="RESULT")

    def process(self):
        while True:
            self.get()  # 获取最新价格
            print("获取最新价格的结果如下：")
            print("历史数据:%s" % self.his_data)
            print("最新数据：%s" % self.rt_data)
            print("收益率计算结果:%s" % self._list)
            self.tot_value = float(self.bfa.futures_account()['totalWalletBalance'])  # 合约账户usdt余额
            print("获取的账户余额数据:%s" % self.tot_value)
            print("此时的持仓情况：%s" % self.position)
            if self.position == 0:
                if self.tot_value != 0:
                    if self.StrongUpSignal_V3:
                        self.signal_record.append('SU')
                        self.place_maker_order(side='BUY')
                        time.sleep(60 * 13)  # 13min后检查挂单是否成交
                        if self.trade:
                            self.position = 1
                    if self.WeakUpSignal_V3:
                        self.signal_record.append("WU")
                        self.place_maker_order(side='BUY')
                        time.sleep(60 * 13)  # 13min后检查挂单是否成交
                        if self.trade:
                            self.position = 1
                    if self.StrongDownSignal:
                        self.signal_record.append("SD")
                        self.place_maker_order(side='SELL')
                        time.sleep(60 * 13)  # 13min后检查挂单是否成交
                        if self.trade:
                            self.position = -1
                    if self.WeakDownSignal:
                        self.signal_record.append("WD")
                        self.place_maker_order(side='SELL')
                        time.sleep(60 * 13)  # 13min后检查挂单是否成交
                        if self.trade:
                            self.position = -1
            else:
                if self.position == 1:
                    if self.LongClosed:
                        self.signal_record.append("LC")
                        self.place_maker_order(side='SELL')
                        time.sleep(60 * 13)  # 13min后检查挂单是否成交
                        if not self.trade:
                            self.liqudation(side='SELL')
                        self.position = 0
                if self.position == -1:
                    if self.ShortClosed:
                        self.signal_record.append("SC")
                        self.place_maker_order(side='BUY')
                        time.sleep(60 * 13)  # 13min后检查挂单是否成交
                        if not self.trade:
                            self.liqudation(side='BUY')
                        self.position = 0
            print("上述条件不符合，等待重连，重新获取数据")
            time.sleep(60 * 13)  # 13min后检查挂单是否成交

    @property
    def LongClosed(self):
        '平多'
        if self.signal_record[-1] in ['SU', 'WU']:
            if self._list[2] <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
                self.ask_price = self.his_data['high'].iloc[1]
                return True
        return False

    @property
    def ShortClosed(self):
        '平空'
        if self.signal_record[-1] in ['SD', 'WD']:
            if self._list[2] > 0:
                self.ask_price = self.his_data['low'].iloc[1]
                return True
        return False

    @property
    def StrongUpSignal_V3(self):
        "强上涨信号"
        if len(self.signal_record) != 0:  # 第一次测试 xx,非第一次不用测试这个
            if self.signal_record[-1] == 'SC' and self._list[1] >= 0 and self._list[2] > self._list[1]:
                self.ask_price = self.rt_data['low']
                return True
            if self._list[1] >= 0 and self._list[2] > self._list[1]:
                self.ask_price = self.rt_data['close']
                return True
        if self._list[1] >= 0 and self._list[2] > self._list[1]:
            self.ask_price = self.rt_data['close']
            return True
        return False

    @property
    def WeakUpSignal_V3(self):
        "弱上涨信号"
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'SC' and self._list[1] >= 0 and self._list[2] > 0:
                self.ask_price = self.rt_data['close']
                return True
        return False

    @property
    def StrongDownSignal(self):
        "强下跌信号"
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and self._list[1] < 0 and self._list[2] < self._list[1]:
                self.ask_price = self.rt_data['close']
                return True
            if self._list[0] < 0 and self._list[1] < self._list[0] and self._list[2] < self._list[1]:
                self.ask_price = self.rt_data['close']
                return True
        if self._list[0] < 0 and self._list[1] < self._list[0] and self._list[2] < self._list[1]:
            self.ask_price = self.rt_data['close']
            return True
        return False

    @property
    def WeakDownSignal(self):
        '弱下跌信号'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and self._list[1] < 0 and self._list[
                2] < 0:
                self.ask_price = self.rt_data['close']
                return True
            if self._list[0] < 0 and self._list[1] < 0 and self._list[2] < 0:
                self.ask_price = self.rt_data['close']
                return True
        if self._list[0] < 0 and self._list[1] < 0 and self._list[2] < 0:
            self.ask_price = self.rt_data['close']
            return True


if __name__ == "__main__":
    r_test = RealtimeStgTest()
    r_test.process()
