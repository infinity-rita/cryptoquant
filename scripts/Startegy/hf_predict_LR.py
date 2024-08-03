"""
write by lzy 2023.1.13
加入预测部分，根据历史数据，仅预测收益率情况

预测：1个变量(-1/1)
二分类：1/0
连续预测未来15min的情况

TODO:校准时间
1.x:15min四档行情 -- y：下一时刻收益率  正确率:0.51~0.52
2.x:当前时刻收益率 -- y：下一时刻收益率 正确率:0.50~0.50
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #train_test_split：随机从数据集中按笔记划分训练标签和测试标签
from sklearn.linear_model import LogisticRegression#step1:导入逻辑回归包

test_csv='2021-all-BTC-5m.csv'
data=pd.read_csv(test_csv)

# 1
X=data[['Open','High','Low','Close']]
# 差分计算
X=X-X.shift(1)
X=X[1:]
# 2 
#X=data['Close']-data['Close'].shift(1)
X=X.iloc[:-1]
y=data['Close']/(data['Close'].shift(1))-1
y=y[2:]
# 预测下一时刻的收益率
# 仅预测当天的
average_predict=[]
for i in range(100):
    X1=X[i*15:43200*4+i*15]
    y1=y[i*15:43200*4+i*15]
    Y=[0 if i<=0 else 1 for i in y1]
    X_train , X_test , y_train , y_test = train_test_split(X1 ,Y ,train_size = 0.85)
    model = LogisticRegression()                       # step2 创建模型：逻辑回归
    model.fit(X_train , y_train) 
    correction=model.score(X_test , y_test)#X_test：测试数据特征；y_test：测试数据标签
    average_predict.append(correction)
    print("模型准确率：%s"%correction)
print("平均准确率：%s"%np.average(average_predict))