
"""
对金融时间序列进行Garch建模
注：1.pyflux包版本过旧，采用arch包构建GARCH模型
参考：https://blog.csdn.net/weixin_42163563/article/details/122114569
https://blog.csdn.net/liuxoxo/article/details/85218135

该方法适合
1.比较好的模型：方差小，偏差小 2.适合波动性和趋势较小的标的 
"""

from arch import arch_model # arch_model 默认建立 GARCH(1,1) 模型
import numpy as np
import pandas as pd
from datetime import datetime
from pandas_datareader import data

class GARCH(object):
    def __init__(self,secu_data):
        data_all=secu_data.astype("float64")  # 原数据类型为str，更改数据类型
        self.returns = pd.DataFrame(np.diff(np.log(data_all['Close'].values))) # 结果与 np.log(close/preclose) 相同
        self.returns.columns=['returns']
        am=arch_model(self.returns**2)
        self.model=am.fit(update_freq=0) # update_freq=0表示不输出中间结果，只输出最终结果
        # print(self.model.summary())

    def process(self):
        # 预测
        forecasts=self.model.forecast(horizon=5) # 一般预测一步，由t时刻预测t+1时刻的波动率，但是5步的结果更好一点（如果数据的频率为5min，则是预测[x+5min,x+10min,x+15min,x+20min,x+25min]
        #print(forecasts.variance.dropna().head()) # 预测未来5天，其预测方差
        return forecasts
    
    def plot(self):
        # 标准残差或波动率的图形化展示
        fig=self.model.plot()
        fig.show()
        # 年化波动率
        #fig=self.model.plot(annualize='D')
        #fig.show()
        #fig=self.model.plot(scale=360)
        #fig.show()

    def run(self):
        self.process()
        self.plot()
        
if __name__ == '__main__':
    secu_data=data.DataReader('005380','naver', datetime(2006,1,1), datetime(2007,1,10)) # 韩国现代汽车
    G=GARCH(secu_data) # 如果secu_data为5min的频次，则计算5min的波动率；日频数据则计算每天的波动率
    G.run()