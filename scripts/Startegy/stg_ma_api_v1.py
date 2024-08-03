"""
write by rita 1.25.2023
使用ma均线策略，核心因子 max(abs(ma7-ma25))
最高点，若ma7>ma25,则做空，否则做多
这里包括滚仓操作哈

历史交易数据：存在本地 tradingresult.csv
TODO：1.运维，确保接口连接稳定，服务器稳定 2.每一单操作发送给手机,qq or wechat

加qq设置 交易提醒哈

"""
import datetime
import time

import numpy as np
import pandas as pd
from binance.client import Client

from tools.robot import send_msg


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
        "检查接口连接情况"
        self.bfa = self.connect
        self.his_data = self.get_historical_kline_data(symbol="ETHUSDT", interval="5m",
                                                       start_time=round((time.time() - 5 * 60 * 100 - 1) * 1000))
        self.his_data['ma7'] = self.his_data['open'].rolling(7).mean()
        self.his_data['ma25'] = self.his_data['open'].rolling(25).mean()
        self.his_data['diff'] = self.his_data['ma25'] - self.his_data['ma7']
        self.his_data['diff'] = self.his_data['diff'].fillna(1)
        self.his_data['abs_diff'] = abs(self.his_data['diff'])
        self.his_data = self.his_data[
            ['StartTime', 'open', 'high', 'low', 'close', 'volume', 'EndTime', 'Amount', 'ma7',
             'ma25', 'diff', 'abs_diff']]
        self.test = self.his_data['StartTime'].str.split(":", expand=True)
        self.his_data['min_time'] = self.test[1].astype(int)
        # 变15min线
        # self.his_data = self.his_data[self.his_data['min_time'] % 15 == 0]
        self.his_data.loc[self.his_data['abs_diff'] <= 5, 'abs_diff'] = 0  # 阈值暂时用5
        self.his_data.to_csv("ma_trading_result.csv")  # 存储数据
        return self.his_data

    def process_action(self):
        action = self.get_his_highest_data(self.his_data)
        # 结合ma25与ma7的大小，判断进行哪个方向的操作
        if action and self.his_data['diff'].iloc[-1] > 0:  # ma25>ma7
            return 1
        if action and self.his_data['diff'].iloc[-1] < 0:  # ma7>ma25
            return -1
        return 0

    def get_his_highest_data(self, df):
        "计算历史峰值, 需要在本地存储历史最大值,要在下一个到了之后，才能确定现在是否是差异最大的部分"
        action = False
        his_highest = self.cal_his_highest_diff(df=df, his_price=0)
        print(his_highest)
        if len(his_highest) != 0 and his_highest['abs_diff'].iloc[-1] != df['abs_diff'].iloc[-1]:
            # TODO:还有需要修改的，以及断联的问题
            action = True
        his_highest.to_csv("ma_his_highest.csv")
        return action

    def cal_his_highest_diff(self, df, his_price):
        his_highest = []
        new_data = []
        label = 0
        for i in range(1, len(df)):
            new_data.append(df['abs_diff'].iloc[i])
            if df['abs_diff'].iloc[i] != 0:
                his_highest_price_index = np.argmax(new_data)
                if new_data[his_highest_price_index] > his_price:
                    his_price = new_data[his_highest_price_index]
                    if len(his_highest) != 0:
                        his_highest.pop()
                his_index = his_highest_price_index + label
                if [new_data[his_highest_price_index], df['open'].iloc[his_index]] in his_highest:
                    his_highest.pop()
                his_highest.append([new_data[his_highest_price_index], df['open'].iloc[his_index]])
            else:
                new_data = []
                his_highest = []
                label = i + 1
                if label < len(df):
                    his_price = df['abs_diff'].iloc[i + 1]
        his_highest = pd.DataFrame(his_highest, columns=['abs_diff', 'open'])
        return his_highest

    def get_latest_kline_data(self, symbol, interval):
        '调用接口,通过api获取最新的数据'
        data = self.bfa.futures_continous_klines(pair=symbol, contractType="PERPETUAL", interval=interval, limit=1)[0][
               :8]  # 获取最新的5min klines数据
        data_list = {'StartTime': timetodate(data[0]), 'open': float(data[1]), 'high': float(data[2]),
                     'low': float(data[3]), 'close': float(data[4]), 'volume': float(data[5]),
                     'EndTime': timetodate(data[6]), 'amount': float(data[7])}
        return data_list

    def get_historical_kline_data(self, symbol, interval, start_time):
        '获取历史时刻kline数据'
        # 香港ip连
        data = self.bfa.futures_historical_klines(symbol=symbol, interval=interval,
                                                  start_str=start_time)  # 获取历史的5min klines数据
        data = np.array(data)[:, :8]
        data_list = pd.DataFrame(data,
                                 columns=['StartTime', 'open', 'high', 'low', 'close', 'volume', 'EndTime', 'Amount'])
        data_list = data_list.astype('float64')
        data_list['StartTime'] = data_list['StartTime'].apply(timetodate)
        data_list['EndTime'] = data_list['EndTime'].apply(timetodate)
        return data_list

    def place_maker_order(self, side):
        '做maker,建仓/平仓,limit下单,以最优价格成交'
        self.tot_value = float(self.bfa.futures_account()['totalWalletBalance']) / 2
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

    def liqudation(self, side):
        '做taker,强平'
        self.positionAmt = float(self.bfa.futures_position_information(symbol='ETHUSDT')[0]['positionAmt'])
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="MARKET",
                                                             quantity=abs(self.positionAmt), newOrderRespType="RESULT")

    def timesync(self):
        '时间同步 -- 建仓'
        u_time = timetodate(time.time() * 1000)[-8:].replace(':', '')
        p_time = self.his_data['StartTime'].iloc[-1][-8:].replace(':', '')
        interval_min = int(p_time[:4]) + 5 - int(u_time[:4]) - 1  # 5min同步用5，15min同步用15
        interval_second = 60 - int(u_time[-2:])
        if interval_min >= 0:
            time.sleep(60 * interval_min + interval_second + 1)

    def process(self):
        while True:
            # self.qqbot("=================策略运行时间:" + timetodate(time.time() * 1000) + "=================")
            self.get()  # 获取数据
            self.positionAmt = float(self.bfa.futures_position_information(symbol='ETHUSDT')[0]['positionAmt'])
            if self.positionAmt > 0:
                self.position = 1
            elif self.positionAmt < 0:
                self.position = -1
            else:
                self.position = 0
            print("获取的历史数据:")
            print(self.his_data[['StartTime', 'open', 'diff', 'abs_diff', 'EndTime']].iloc[-3:])
            self.tot_value = float(self.bfa.futures_account()['totalWalletBalance'])  # 合约账户usdt余额
            print("获取的账户余额数据:%s" % self.tot_value)
            print("此时的持仓情况：%s" % self.position)
            action = self.process_action()  # 是否建仓
            print("因子abs_diff:%s" % self.his_data['abs_diff'].iloc[-1])
            print("建仓情况:%s" % action)
            if self.position == 0 and self.tot_value != 0:  # 有钱开仓
                if action == 1:  # 开多
                    self.ask_price = self.his_data['open'].iloc[-1]
                    self.place_maker_order(side='BUY')
                    print("时间同步，等待约5min检验成交")
                    self.timesync()
                    if self.trade:
                        self.position = 1
                        print("做多挂单成交")
                        current_time = datetime.datetime.now()
                        # self.qqbot(str(current_time) + " 做多，开单价格：$" + str(self.ask_price))
                    else:
                        print("做多挂单未成交")
                    continue
                if action == -1:  # 开空
                    self.ask_price = self.his_data['open'].iloc[-1]
                    self.place_maker_order(side='SELL')
                    print("时间同步，等待约5min检验成交")
                    self.timesync()
                    if self.trade:
                        self.position = -1
                        print("做空挂单成交")
                        current_time = datetime.datetime.now()
                        # self.qqbot(str(current_time) + " 做空，开单价格：$" + str(self.ask_price))
                    else:
                        print("做空挂单未成交")
                    continue
            else:  # 先平仓随后换仓
                if self.position == 1 and action == 1:  # 先平多后开多
                    self.liqudation(side='SELL')
                    self.ask_price = self.his_data['open'].iloc[-1]
                    self.place_maker_order(side='BUY')
                    print("时间同步，等待约5min检验成交")
                    self.timesync()
                    if not self.trade:
                        self.position = 0
                        print("平多成功，但开多挂单未成交")
                        current_time = datetime.datetime.now()
                        pnl = self.bfa.futures_income_history(symbol='ETHUSDT', incomeType='REALIZED_PNL')[-1]['income']
                        # self.qqbot(str(current_time) + " 平多，收益率：" + pnl / self.tot_value * 100)
                    else:
                        print("平多并开多挂单成交")
                    continue
                if self.position == 1 and action == -1:  # 先平多后开空
                    self.liqudation(side='SELL')
                    self.ask_price = self.his_data['open'].iloc[-1]
                    self.place_maker_order(side='SELL')
                    print("时间同步，等待约5min检验成交")
                    self.timesync()
                    if not self.trade:
                        self.position = 0
                        print("平多成功，但开空挂单未成交")
                        current_time = datetime.datetime.now()
                        pnl = self.bfa.futures_income_history(symbol='ETHUSDT', incomeType='REALIZED_PNL')[-1]['income']
                        # self.qqbot(str(current_time) + " 平多，收益率：" + pnl / self.tot_value * 100)
                    else:
                        self.position = -1
                        print("平多并开空挂单成交")
                    continue
                if self.position == -1 and action == 1:  # 先平空后开多
                    self.liqudation(side='BUY')
                    self.ask_price = self.his_data['open'].iloc[-1]
                    self.place_maker_order(side='BUY')
                    print("时间同步，等待约5min检验成交")
                    self.timesync()
                    if not self.trade:
                        self.position = 0
                        print("平空成功，但开多挂单未成交")
                        current_time = datetime.datetime.now()
                        pnl = self.bfa.futures_income_history(symbol='ETHUSDT', incomeType='REALIZED_PNL')[-1]['income']
                        # self.qqbot(str(current_time) + " 平空，收益率：" + pnl / self.tot_value * 100)
                    else:
                        self.position = 1
                        print("平空并开多挂单成交")
                    continue
                if self.position == -1 and action == -1:  # 先平空后开空
                    self.liqudation(side='BUY')
                    self.ask_price = self.his_data['open'].iloc[-1]
                    self.place_maker_order(side='SELL')
                    print("时间同步，等待约5min检验成交")
                    self.timesync()
                    if not self.trade:
                        self.position = 0
                        print("平空成功，但开空挂单未成交")
                        current_time = datetime.datetime.now()
                        pnl = self.bfa.futures_income_history(symbol='ETHUSDT', incomeType='REALIZED_PNL')[-1]['income']
                        # self.qqbot(str(current_time) + " 平空，收益率：" + pnl / self.tot_value * 100)
                    else:
                        print("平空并开空挂单成交")
                    continue
            print("================上述条件不符合，等待信号出现，检查接口连接情况，重新获取数据================")
            print("进行时间同步......")
            self.timesync()

    def qqbot(self, info):
        resp_dict = {}
        resp_dict['msg_type'] = 'group'
        resp_dict['number'] = 648887836
        resp_dict['msg'] = info
        send_msg(resp_dict)
        return


if __name__ == "__main__":
    r_test = RealtimeStgTest()
    r_test.process()
