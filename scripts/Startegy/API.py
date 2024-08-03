# 仅test binance api-futures FUTURES_DATA_URL = 'https://fapi.binance.{}/futures/data'
# websocket待调整/补充，
# TODO:1.10测试本脚本内全部可能被调用到的函数，测试通过

api_key="dxFzobejrXqyKYRjE3SB341m7w4oj18Y8c1hl9RRbibT0YCSPaOV6Y1JANnHsQPc" 
api_secret="2BaxckpFPF4uLQ8psXkpukjQA8GPVXbgPvWnO9kDakRosqqLVoVC1oSMjjmGmp2j" 

from binance.client import Client
import time
my_account=Client(api_key=api_key,api_secret=api_secret) # 设香港代理ip连接 futures_data_url

# 测试连接：
my_account.futures_ping() # 返回{} 成功连接
# 检查系统和本地时间是同步的，无需检查
def timetodate(timestamp):
    timeArray = time.localtime(timestamp / 1000)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime
# round(time.time()*1000)可以将timestamp精确到秒

# 1.获取实时/最新价格
lastst_price=my_account.futures_symbol_ticker(symbol='BTCUSDT') # 'ETHUSDT' 返回的是当前时刻的价格
# 返回：{'symbol': 'BTCUSDT', 'price': '17194.30', 'time': 1673314883766}
# 1) 获取历史数据
start_time=1673254300000 # Start date string in UTC format or timestamp in milliseconds [八个非零，5个零]
his_data=my_account.futures_historical_klines(symbol="BTCUSDT",interval="15m",start_str=start_time,limit=500)
# TODO:四个价格都是字符串格式，需要改类型
# 2.隔固定时间获取数据
# 获取5min连续klines数据：1.获取系统时间 2.系统时间+5min，提前1000ms读取数据，如果不设定startTime和endTime，返回的数据为最新的klines数据
param={
    "pair":"BTCUSDT",
    "contractType":"PERPETUAL",
    "interval":"5m"
}
con_klines_data=my_account.futures_continous_klines(pair="BTCUSDT",contractType="PERPETUAL",interval="5m") # 间隔
# 返回数据的时间戳: '2023-01-10 09:35:00' 精确到分钟，非秒
# 返回值：list of OHLCV values (Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore)
# 隔一段时间获取系统数据，用于校正本地时间，进行同步

# 3.发送做单post信息，币安端接收，测试做单是否成功///强平
# 1) 发送订单：【设为逐仓/双向持仓】time.sleep(5min)
param={
    "symbol":"BTCUSDT",
    "side":"BUY", # SELL
    "type":"LIMIT",
    "positionSide":"LONG", # LONG/SHORT根据具体情况设
    "timestamp":1673254300000,
    "newOrderRespType":"RESULT" # 成交结果将被立刻返回
}
place_order_res=my_account.futures_create_order(params=param)
# 返回值包括：orderId等
# 2）检查订单状态：
param={
    "symbol":"BTCUSDT",
    "orderId":"", # 从上一个post请求的返回值获得，[可不发送此参数]
    "timestamp":1673254300000
}
check_res=my_account.futures_get_order(params=param)
# 3）如果返回订单未成交，以实时价格强平（限账户持仓不为0的情况）
# liquidation_order 强制平仓清算
my_account.futures_liquidation_orders()
# 是否通过检查order的方式，吃单？ 直接强平的可能亏损？


# 4.开单，实时开单利率获取等
# 1）没有成交的订单要及时取消：
param={
    "symbol":"BTCUSDT",
    "orderId":"", # 从上一个post请求的返回值获得, [可不发送此参数]
    "timestamp":1673254300000
}
cancel_order=my_account.futures_cancel_order(params=param)

# 2）实时检查期货账户余额
account_balance=my_account.futures_account_balance()


# 其余常用方法
# 1）调整合约杠杆
param={
    "symbol":"BTCUSDT",
    "leverage":3, # 从1-125，int，可选修改为3x
    "timestamp":1673254300000
}
change_leverage=my_account.futures_change_leverage(params=param)

# 2）合约实盘收益情况
active_pnl=my_account.futures_income_history() # 获取某一时刻的收益，就加入timestamp
# 返回：[] ？
# 3) socket stream获得listenkey
listen_key=my_account.futures_stream_get_listen_key()
# 返回的listen_key: 'HwgkAdJfRXzlESwQzw2wKRC7DSnnHzzLtDhE9tHOuaSnFaEpStmcBGcxBtroyUI1'
# websocket stream user data 价格/交易订单更新

# a.listenkey保持可用，最长可保持24小时
my_account.futures_stream_keepalive(listenKey=listen_key)
# 返回:{} ?
# b.关闭stream，重新启socket
my_account.futures_stream_close(listenKey=listen_key)
# 返回:{} ?
# 3）合约账户历史交易情况
my_account.futures_get_all_orders() #没有订单，返回为空
# 获得所有order情况，active,canceled,filled