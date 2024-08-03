"""
测试：修改后的策略12.27.2022
回测部分 -- 永续合约 
V1版本，暂不修改，用于存档
"""
import time
import datetime
import numpy as np
import pandas as pd

class OptionVolatility(object):
    def __init__(self, data, type_, freq, data_type):
        self.close = data['Close'] # 四个指标都要用到，改为 self.data=data
        self.high=data['High']
        self.low=data['Low']
        self.time=data['TimeStamp']
        self.type = type_
        self.freq = freq
        self.freqAdjust()  # 调整频率
        self.data_type = data_type
        self.avgprice=round((self.high+self.low)/2,2)
        returns = self.Volatility
        vol_rate=(self.high-self.low)/self.low # 波动率
        self.result=pd.DataFrame({'tradingtime':self.time,'close':self.close,'low':self.low,'high':self.high,'avgprice':self.avgprice,'volatility':returns,'vol_rate':vol_rate}) # 最终要输出的结果
        self.result['volatility']=self.result['volatility'].fillna(0)
        self.result.loc[self.result['avgprice'].isnull(),'avgprice']=round((self.result.loc[self.result['avgprice'].isnull(),'low']+self.result.loc[self.result['avgprice'].isnull(),'high'])/2,2)
        self.result.to_csv("result.csv")
        self.trend = self.CatchTrendStg()  # 换策略

    def timetodate(self,timestamp):
        timeArray = time.localtime(timestamp/1000)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        return otherStyleTime
    
    def process(self,cash):
        self.signal_record=[]# 记录每一时刻的信号
        position=0 # 最开始未开仓
        self.trading_result={}
        self.ask_price=0.0 # 最开始不竞价
        alltradeqty=0.0
        tot_value=cash
        tot_roi=0.0 # 总回报情况
        print("=============" + self.data_type + "_" + self.freq + "策略开始运行=============")
        for i in range(len(self.result)-3): # 改成while,不断获取数据
            vol_list=self.result.iloc[i:i+3,:]
            # 判断是啥信号，挂单撮合阶段
            pnl=0.0 # 每单损益初始置零，每单收益率初始置零
            roi=0.0
            if position==0 and cash!=0:
                if self.StrongUpSignal(vol_list):
                    self.signal_record.append('SU')
                    #挂单未成交，position不改变
                    alltradeqty,cash,pnl=self.bidAction(i,cash)
                    if alltradeqty!=0:
                        position=1
                elif self.WeakUpSignal(vol_list):
                    self.signal_record.append("WU")
                    alltradeqty,cash,pnl=self.bidAction(i,cash)
                    if alltradeqty!=0:
                        position=1
                elif self.StrongDownSignal(vol_list):
                    self.signal_record.append("SD")
                    alltradeqty,cash,pnl=self.bidAction(i,cash)
                    if alltradeqty!=0:
                        position=-1
                elif self.WeakDownSignal(vol_list):
                    self.signal_record.append("WD")
                    alltradeqty,cash,pnl=self.bidAction(i,cash)
                    if alltradeqty!=0:
                        position=-1
                else:
                    #if i>=24*4 and self.result['vol_rate'].iloc[:i].mean()<0.01:
                        # 在这里判断波动率的情况，太低则切换策略，算一段时间的波动情况 24*4， 1天的平均波动率<1%
                    #    self.ask_price= vol_list['avgprice'].iloc[2]
                    #    alltradeqty, cash, pnl = self.bidAction(i, cash)
                    #    print('yes')
                    #    if alltradeqty != 0:
                    #        position = self.trend[i] #1对应开多，-1对应开空
                    #else:
                    pass
            elif self.LongClosed(vol_list) and position==1:
                self.signal_record.append("LC")
                position=0
                alltradeqty,cash,pnl=self.CloseAction(i,position,alltradeqty)
            elif self.ShortClosed(vol_list) and position==-1:
                self.signal_record.append("SC")
                position=0
                alltradeqty,cash,pnl=self.CloseAction(i,position,alltradeqty)
            else:
                pass
            # orderbook成交状态更新
            if cash!=0 and pnl!=0:
                roi=pnl/(cash-pnl)*100
                tot_roi=(cash-tot_value)/tot_value*100
            basic_return = (self.result['close'].iloc[i+2]-self.result['close'].iloc[0])/self.result['close'].iloc[0]*100
            pos_value=(tot_roi/100+1)*tot_value
            orderbook={'portfolio':'BTC','position':position,'tradeprice':self.ask_price,'alltradeqty':alltradeqty,'askprice':self.ask_price,
                            'cash':cash,'pnl':pnl, 'roi(%)':roi,'tot_roi(%)':tot_roi,'portfolio_price':self.result['close'].iloc[i+2],'basic_return(%)':basic_return,'pos_value':pos_value}
            current_time=self.timetodate(self.result['tradingtime'].iloc[i])
            if len(self.trading_result):
                res=pd.DataFrame(orderbook,index=[current_time])
                self.trading_result=pd.concat([self.trading_result,res])
            else:
                self.trading_result=pd.DataFrame(orderbook,index=[current_time])
        print("=============" + self.data_type + "_" + self.freq + "策略运行结束，交易结果如下所示=============")
        print(self.trading_result)

    def CatchTrendStg(self):
        # 可以进一步修改哈
        vol_20x_min = self.result['volatility'].rolling(20).mean()  # 最近5小时的平均波动率 -- 15min线对应是100min
        vol_4x_min = self.result['volatility'].rolling(4).mean()  # 最近1小时的平均波动率 -- 15min线对应是
        vol_100x_min = self.result['volatility'].rolling(100).mean()  # 25小时线
        res = []
        for i in range(1, len(vol_20x_min)):
            if vol_20x_min[i] > vol_4x_min[i] and vol_4x_min[i - 1] > vol_4x_min[i] and \
                    vol_100x_min[i] > vol_20x_min[i] and vol_20x_min[i - 1] > vol_20x_min[i]:
                res.append(-1)
            else:
                res.append(1)
        self.res = res
        return res
        
    def bidAction_Old(self,num,cash):
        alltradeqty=0.0
        pnl = 0.0
        if self.result['high'].iloc[num+3]>=self.ask_price and self.result['low'].iloc[num+3]<=self.ask_price: # 挂单成交
            alltradeqty = cash* (1 - 0.04 / 100)/(self.ask_price)
            cash = 0
            #print("挂单成交!成交价格:$%s"%self.ask_price) 
            return [alltradeqty,cash,pnl]
        else:
            #print('挂单未成交，等待下一个交易机会!')
            return [alltradeqty,cash,pnl]
        
    def bidAction(self,num,cash):
        # 假设挂单全部成交
        pnl = 0.0
        alltradeqty = cash* (1 - 0.04 / 100)/(self.ask_price)
        cash = 0
        #print("挂单成交!成交价格:$%s"%self.ask_price) 
        return [alltradeqty,cash,pnl]
        
    def CloseAction(self,num,position,alltradeqty):
        # 平仓操作，即刻以close价平仓，默认一定成交， pnl包括平仓手续费
        pnl = alltradeqty * (self.ask_price - self.trading_result['askprice'].iloc[num-1])*(self.trading_result['position'].iloc[num-1]-position) - self.ask_price*alltradeqty*0.04/100
        cash = pnl +  self.trading_result['alltradeqty'].iloc[num-1] * self.trading_result['tradeprice'].iloc[num-1]
        return [alltradeqty,cash,pnl]
    
    def StrongUpSignal(self,vol_list):
        # 强上涨信号
        returns=vol_list['volatility']
        if returns.iloc[0]>=0:
            if returns.iloc[2]>returns.iloc[1] and returns.iloc[1]>returns.iloc[0] and vol_list['close'].iloc[2]>vol_list['avgprice'].iloc[2]: # 连续上涨，且上涨幅度越来越大
                self.ask_price=vol_list['avgprice'].iloc[2] # 以长期横盘价挂单(暂以最后时刻的均价挂单)
                #print('======出现强上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
                return True
        return False
    
    def WeakUpSignal(self,vol_list):
        # 弱上涨信号
        returns=vol_list['volatility']
        if returns.iloc[0]>=0 and returns.iloc[1]>=0 and returns.iloc[2]>=0: # 震荡行情，但不下跌
            self.ask_price=vol_list['avgprice'].iloc[2] # 取三个时段的平均价挂单做多(暂以最后时刻的均价挂单)
            #print('======出现弱上涨信号,挂单,价格:$'+str(self.ask_price)+',准备策略做多======')
            return True
        return False
    
    def LongClosed(self,vol_list):
        # 平多
        returns=vol_list['volatility']
        if self.signal_record[-1] in ['SU','WU'] and returns.iloc[2]<0:
            self.ask_price=vol_list['close'].iloc[2] 
            #print('======平多,平仓价格:$'+str(self.ask_price)+'======')
            return True
        return False
    
    def StrongDownSignal(self,vol_list):
        # 强下跌信号
        returns=vol_list['volatility']
        if len(self.signal_record)!=0:
            if self.signal_record[-1]=='LC' and returns.iloc[0]<0 and returns.iloc[1]<returns.iloc[0]:
                self.ask_price=vol_list['close'].iloc[1] 
                #print('======出现强下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
        elif returns.iloc[0]<0 and returns.iloc[1]<returns.iloc[0] and returns.iloc[2]<returns.iloc[1]:
            self.ask_price=vol_list['close'].iloc[2] 
            #print('======出现强下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
            return True
        else:
            pass
        return False
    
    def WeakDownSignal(self,vol_list):
        # 弱下跌信号
        returns=vol_list['volatility']
        if len(self.signal_record)!=0:
            if self.signal_record[-1]=='LC' and returns.iloc[0]<0 and returns.iloc[1]<0:
                self.ask_price=vol_list['avgprice'].iloc[1] 
                #print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
                return True
        elif returns.iloc[0]<0 and returns.iloc[1]<0 and returns.iloc[2]<0:
            self.ask_price=vol_list['avgprice'].iloc[2] 
            #print('======出现弱下跌信号,挂单,价格:$'+str(self.ask_price)+',准备策略做空======')
            return True
        else:
            pass
        return False
    
    def ShortClosed(self,vol_list):
        # 平空
        returns=vol_list['volatility']
        if self.signal_record[-1] in ['SD','WD'] and returns.iloc[2]>0: #一旦下跌趋势中止就赶紧跑，不考虑什么波动率最大了
            self.ask_price=vol_list['low'].iloc[1] 
            #print('======平空,平仓价格:$'+str(self.ask_price)+'======')
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
        else:
            pass

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
    i=np.argmax((np.maximum.accumulate(return_list)-return_list)) # 最大回撤结束的位置
    if i==0:
        j=0
    else:
        j=np.argmax(return_list.iloc[:i]) # 回撤开始的位置
    print(i,j)
    print(return_list.iloc[j],return_list.iloc[i])
    return (return_list.iloc[j]-return_list.iloc[i])/(return_list.iloc[j])*100
secu_data_monthly = pd.read_csv("2022-yearly-newest-BTC-5m.csv")

secu_data_daily = pd.read_csv("BTCUSDT-5m-2022-11-23.csv")
secu_data = {"monthly": secu_data_monthly}
import math
# 测试1个月和1天的的数据：
for k, v in secu_data.items():
    ST = OptionVolatility(v, type_='history', freq="15min", data_type=k)
    ST.run(cash=10000)
    res=ST.trading_result
    res.to_csv("trading_result.csv")
    print('最大回撤是：%s'%MaxDrawdown(res['pos_value']))


