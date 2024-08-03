# coding=utf-8
import math
import tushare as ts
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import talib
import pandas as pd
from datetime import datetime, date
import seaborn as sns  
sns.set(style="darkgrid", palette="muted", color_codes=True) 
from scipy import stats,integrate

sns.set(color_codes=True)

matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei']
ts.set_token('0a输入token码')#获取地址 https://tushare.pro/register?reg=385920
pro = ts.pro_api()
#读取数据
star="20040602"
end="20200905"
df = pro.index_daily( ts_code='000300.SH', start_date=star, end_date=end)
df=df.sort_index(ascending=False)
df.index=pd.to_datetime(df.trade_date,format='%Y-%m-%d')#设置日期索引
df_close=df.close/df.close[0]#计算净值

df.trade_date=None
for i in range(20,len(df.index)):#计算20天的波动率
    df.trade_date[i]=np.std(np.log(df_close[i-20:i] /df_close[i-20:i].shift(-1)))*np.sqrt(252)*100#波动率计算
df1=df.trade_date.dropna()

"从这里开始的函数是重要的"
def CL_fun(t,T,df1=df1,df_close=df_close):
    df1=pd.Series(df1,dtype=float)
    df2= talib.MA(pd.Series(pd.Series(df1 ,dtype=np.float) ), timeperiod=T)
    sig=pd.Series(0,df1.index)
    for i in range(math.ceil(T/t), math.floor(len(df1)/t)-1):
        if df1[i*t+t]>df2[i*t+t] and df1[i*t]<df1[i*t+t]:#这里可以作为检验，即去掉if  查看波动
            for j in range(i*t+1,(i+1)*t+1):
                sig[j]=1
    df_close=df_close.sort_index()
    ret=(df_close-df_close.shift(1))/df_close.shift(1)
    ret1=ret.tail(len(df1)).sort_index()*sig
    cum=np.cumprod(ret1+1).dropna()
    cum=cum.tail(len(cum)-T-2)

    def Tongji(cum):
        cum=cum.sort_index()
        NH=(cum[-1]-1)*100*252/len(cum.index)
        BD=np.std(np.log(cum/cum.shift(-1)))*np.sqrt(252)*100
        SR=(NH-4)/BD
        return_list=cum
        MHC=((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)).max()*100
        print("年化收益率：{:.2f}%:，年化夏普率：{:.2f},波动率为：{:.2f}%,最大回撤：{:.2f}%".format( NH,SR,BD,MHC))

    return Tongji(cum),cum
if __name__=="__main__":
############################################################################
    plt.plot( CL_fun(10,100,)[1],label="t=10,T=100",color='r',linestyle='-')
    plt.plot( CL_fun(30,100,)[1],label="t=30,T=100",color='g',linestyle='-')
    plt.plot( CL_fun(20,110,)[1],label="t=20,T=110",color='b',linestyle='-')
    plt.plot( CL_fun(20,80,)[1],label="t=20,T=80",color='y',linestyle='-')
    plt.plot( CL_fun(20,130,)[1],label="t=20,T=130",color='m',linestyle='-')
    plt.title("2005-2010年 反向—移动波动率策略策略净值走势图 ")
    plt.legend()        


