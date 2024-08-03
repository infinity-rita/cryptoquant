"""
# 1.10 TODO：1.优化策略的判断条件 2.私募 1.11 ok，10u调试接口(强平价格部分修改;强平信号到来前后修改)
# 根据V10的版本进行修改

1.11 进入测试阶段
1) 查一下maker和taker两者触发的条件各是什么情况
2) 检查挂单价格的时间先后问题，数据来临的先后问题，和ask_price设置的先后问题
3) 关于实时强制平仓和止盈止损的设置？关于提前获取return? test v2
a.止盈止损？
b.实时强制平仓用：市价吃单
c.提前获取return:15min(14min);10min(7min)

v1:5m快速测试 -- 1小时内是否有误差，anyway，报一下实时获取数据的时间,timestamp

存在的问题：
1.海外代理节点连接不稳，存在反复重连的情况  try - except -- 改用10min测试 - 10/7
2.程序需要部署在云服务器上一直运行，选择香港节点

3.保证金率不充足，可以减少一半本金，2x杠杆，有一定的保证金率

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
        self.position = 0  # 最开始无持仓
        try:
            self.bfa = self.connect
            self.bfa.futures_ping()
        except:
            time.sleep(5)
            self.bfa = self.connect
            self.bfa.futures_ping()

    @property
    def connect(self):
        # 定时15min重连，确保和服务器保持连接
        api_key = "dxFzobejrXqyKYRjE3SB341m7w4oj18Y8c1hl9RRbibT0YCSPaOV6Y1JANnHsQPc"
        api_secret = "2BaxckpFPF4uLQ8psXkpukjQA8GPVXbgPvWnO9kDakRosqqLVoVC1oSMjjmGmp2j"
        bfa = Client(api_key=api_key, api_secret=api_secret)
        return bfa

    def get(self):
        "获取计算需要的数据，断连则重连"
        try:
            self.bfa.futures_ping()
        except:
            self.bfa = self.connect
        self.his_data = self.get_historical_kline_data(symbol="ETHUSDT", interval="15m",
                                                       start_time=round((time.time() - 15 * 60 * 4 - 1) * 1000))
        self._list = [self.his_data['return'].iloc[-3], self.his_data['return'].iloc[-2],
                      self.his_data['return'].iloc[-1]]  # 策略滚动测试部分

    def get_historical_kline_data(self, symbol, interval, start_time):
        '获取历史时刻kline数据,limit 3 计算临近2time return'
        # 香港ip连
        data = self.bfa.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_time)
        data = np.array(data)[:, :8]
        data_list = pd.DataFrame(data,
                                 columns=['StartTime', 'open', 'high', 'low', 'close', 'volume', 'EndTime', 'Amount'])
        data_list = data_list.astype('float64')
        data_list['StartTime'] = data_list['StartTime'].apply(timetodate)
        data_list['EndTime'] = data_list['EndTime'].apply(timetodate)
        data_list['return'] = np.log(data_list['open'] / data_list['open'].shift(1))
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
        "检查订单是否成交"
        order_status = self.bfa.futures_get_order(symbol="ETHUSDT", orderId=self.place_order_res['orderId'])['status']
        if order_status != 'NEW':  # 已成交或已取消
            return True
        else:
            self.bfa.futures_cancel_all_open_orders(symbol="ETHUSDT")  # 未成交，取消全部订单，避免挂单冲突
            return False

    def liqudation_V1(self, side):
        '做taker,强平/maker,止盈止损单'
        if side == 'BUY':
            self.ask_price = self.his_data['low'].iloc[-1]
        if side == 'SELL':
            self.ask_price = self.his_data['high'].iloc[-1]
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="STOP_MARKET",
                                                             priceProtect="TRUE",
                                                             closePosition="TRUE", workingType='CONTRACT_PRICE',
                                                             stopPrice=self.ask_price, newOrderRespType="RESULT")

    # V2 市价强平止盈止损
    def liqudation(self, side):
        '做taker,强平'
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="MARKET",
                                                             quantity=abs(self.positionAmt), newOrderRespType="RESULT")

    def timesync(self):
        '时间同步 -- 建仓'
        u_time = timetodate(time.time() * 1000)[-8:].replace(':', '')
        p_time = self.his_data['StartTime'].iloc[-1][-8:].replace(':', '')
        interval_min = int(p_time[:4]) + 15 - int(u_time[:4]) - 1
        interval_second = 60 - int(u_time[-2:])
        if interval_min >= 0:
            time.sleep(60 * interval_min + interval_second + 1)

    def recvtimesync(self):
        '时间同步 -- 平仓，读14min的实时价格'
        u_time = timetodate(time.time() * 1000)[-8:].replace(':', '')
        p_time = self.his_data['StartTime'].iloc[-1][-8:].replace(':', '')
        interval_min = int(p_time[:4]) + 14 - int(u_time[:4]) - 1
        interval_second = 60 - int(u_time[-2:])
        if interval_min >= 0:
            time.sleep(60 * interval_min + interval_second + 1)

    def process(self):
        while True:
            print("策略运行时间:%s" % timetodate(time.time() * 1000))
            self.positionAmt = float(self.bfa.futures_position_information(symbol='ETHUSDT')[0]['positionAmt'])
            if self.positionAmt > 0:
                self.position = 1
            elif self.positionAmt < 0:
                self.position = -1
            else:
                self.position = 0
            self.tot_value = float(self.bfa.futures_account()['totalWalletBalance'])  # 合约账户usdt余额
            print("获取的账户余额数据:%s" % self.tot_value)
            print("持仓情况:%s" % self.position)
            if self.position == 0:
                if self.tot_value != 0:
                    self.get()  # 获取最新价格
                    print("获取最新价格/历史数据的结果如下：")
                    print(self.his_data[['StartTime', 'open', 'high', 'low', 'EndTime']])
                    if self.StrongUpSignal_V3:
                        self.signal_record.append('SU')
                        self.place_maker_order(side='BUY')
                        print("触发强上涨条件，策略做多，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做多挂单未成交")
                        else:
                            print("做多挂单成交")
                        continue
                    if self.WeakUpSignal_V3:
                        self.signal_record.append("WU")
                        self.place_maker_order(side='BUY')
                        print("触发弱上涨条件，策略做多，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做多挂单未成交")
                        else:
                            print("做多挂单成交")
                        continue
                    if self.StrongDownSignal:
                        self.signal_record.append("SD")
                        self.place_maker_order(side='SELL')
                        print("触发强下跌条件，策略做空，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做空挂单未成交")
                        else:
                            print("做空挂单成交")
                        continue
                    if self.WeakDownSignal:
                        self.signal_record.append("WD")
                        self.place_maker_order(side='SELL')
                        print("触发弱下跌条件，策略做空，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做空挂单未成交")
                        else:
                            print("做空挂单成交")
                        continue
            else:
                if self.position == 1:
                    self.get()
                    print("获取最新价格/历史数据的结果如下：")
                    print(self.his_data[['StartTime', 'open', 'high', 'low', 'EndTime']])
                    print("等待一段时间后计算瞬时收益率情况")
                    self.recvtimesync()
                    open_price = float(self.bfa.futures_symbol_ticker(symbol="ETHUSDT")['price'])
                    self.return_recv = np.log(open_price / self.his_data['open'].iloc[-1])
                    print("获取瞬时价格:%s" % timetodate(time.time() * 1000))
                    print(self.return_recv)
                    if self.LongClosed_V2:
                        self.signal_record.append("LC")
                        self.place_maker_order(side='SELL')
                        print("触发平多条件，策略平多，平仓价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            self.liqudation(side='SELL')
                            print("平多挂单未成交,强制平仓")
                        else:
                            print("平多挂单成交!")
                        continue
                if self.position == -1:
                    self.get()
                    print("获取最新价格/历史数据的结果如下：")
                    print(self.his_data[['StartTime', 'open', 'high', 'low', 'EndTime']])
                    print("等待一段时间后计算瞬时收益率情况")
                    self.recvtimesync()
                    open_price = float(self.bfa.futures_symbol_ticker(symbol="ETHUSDT")['price'])
                    self.return_recv = np.log(open_price / self.his_data['open'].iloc[-1])
                    print("获取瞬时价格:%s" % timetodate(time.time() * 1000))
                    print(self.return_recv)
                    if self.ShortClosed_V2:
                        self.signal_record.append("SC")
                        self.place_maker_order(side='BUY')
                        print("触发平空条件，策略平空，平仓价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            self.liqudation(side='BUY')
                            print("平空挂单未成交,强制平仓")
                        else:
                            print("平空挂单成交!")
                        continue
            print("================上述条件不符合，等待信号出现，检查接口连接情况，重新获取数据================")
            print("进行时间同步......")
            self.timesync()

    @property
    def LongClosed_V2(self):
        '平多'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SU', 'WU']:
                if self.return_recv <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
                    self.ask_price = self.his_data['high'].iloc[-2]
                    return True
        if self.return_recv <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
            self.ask_price = self.his_data['high'].iloc[-2]
            return True
        return False

    @property
    def ShortClosed_V2(self):
        '平空'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SD', 'WD']:
                if self.return_recv > 0:
                    self.ask_price = self.his_data['low'].iloc[-2]
                    return True
        if self.return_recv > 0:
            self.ask_price = self.his_data['low'].iloc[-2]
            return True
        return False

    @property
    def LongClosed(self):
        '平多'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SU', 'WU']:
                if self._list[2] <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
                    self.ask_price = self.his_data['high'].iloc[2]
                    return True
        if self._list[2] <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
            self.ask_price = self.his_data['high'].iloc[2]
            return True
        return False

    @property
    def ShortClosed(self):
        '平空'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SD', 'WD']:
                if self._list[2] > 0:
                    self.ask_price = self.his_data['low'].iloc[2]
                    return True
        if self._list[2] > 0:
            self.ask_price = self.his_data['low'].iloc[2]
            return True
        return False

    @property
    def StrongUpSignal_V3(self):
        "强上涨信号"
        if len(self.signal_record) != 0:  # 第一次测试 xx,非第一次不用测试这个
            if self.signal_record[-1] == 'SC' and self._list[1] >= 0 and self._list[2] > self._list[1]:
                self.ask_price = self.his_data['low'].iloc[-2]
                return True
            if self._list[1] >= 0 and self._list[2] > self._list[1]:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        if self._list[1] >= 0 and self._list[2] > self._list[1]:
            self.ask_price = self.his_data['open'].iloc[-1]
            return True
        return False

    @property
    def WeakUpSignal_V3(self):
        "弱上涨信号"
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'SC' and self._list[1] >= 0 and self._list[2] > 0:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        return False

    @property
    def StrongDownSignal(self):
        "强下跌信号"
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and self._list[1] < 0 and self._list[2] < self._list[1]:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
            if self._list[0] < 0 and self._list[1] < self._list[0] and self._list[2] < self._list[1]:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        if self._list[0] < 0 and self._list[1] < self._list[0] and self._list[2] < self._list[1]:
            self.ask_price = self.his_data['open'].iloc[-1]
            return True
        return False

    @property
    def WeakDownSignal(self):
        '弱下跌信号'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and self._list[1] < 0 and self._list[
                2] < 0:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
            if self._list[0] < 0 and self._list[1] < 0 and self._list[2] < 0:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        if self._list[0] < 0 and self._list[1] < 0 and self._list[2] < 0:
            self.ask_price = self.his_data['open'].iloc[-1]
            return True


if __name__ == "__main__":
    r_test = RealtimeStgTest()
    r_test.process()
