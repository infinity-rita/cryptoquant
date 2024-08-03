# 测试websocket user data stream获取交易订单情况下
# TODO:测试失败
api_key="dxFzobejrXqyKYRjE3SB341m7w4oj18Y8c1hl9RRbibT0YCSPaOV6Y1JANnHsQPc" 
api_secret="2BaxckpFPF4uLQ8psXkpukjQA8GPVXbgPvWnO9kDakRosqqLVoVC1oSMjjmGmp2j" 

from binance.client import Client
import time
my_account=Client(api_key=api_key,api_secret=api_secret) # 设香港代理ip连接 futures_data_url

# 1.生成具有唯一监听密钥的用户数据流，每24小时重新启动连接，生成新的listen key
listen_key=my_account.futures_stream_get_listen_key()
# 返回的listen_key: 'HwgkAdJfRXzlESwQzw2wKRC7DSnnHzzLtDhE9tHOuaSnFaEpStmcBGcxBtroyUI1'

# 为保持listen_key处于活跃状态，需每30分钟发送一次ping，即可延长60min有效期
my_account.futures_stream_keepalive(listenKey=listen_key)

# 2.测试binance websocket端点
# binance futures: wss://fstream.binance.com或wss://fstream-auth.binance.com

# wss://fstream.binance.com
stream1='wss://fstream.binance.com/ws/btcusdt@kline_5m'
stream2='wss://fstream.binance.com/stream?streams=bnbusdt@aggTrade/btcusdt@kline_5m' # stream name:<symbol>@kline_<interval>

# wss://fstream-auth.binance.com
stream1='wss://fstream-auth.binance.com/ws/btcusdt@kline_5m?listenKey=HwgkAdJfRXzlESwQzw2wKRC7DSnnHzzLtDhE9tHOuaSnFaEpStmcBGcxBtroyUI1'

# 订阅流：
# kline streaems 更新速度250ms
# Continuous Contract Kline eg: btcusdt_perpetual@continuousKline_5m