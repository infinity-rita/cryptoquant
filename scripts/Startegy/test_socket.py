# test socket
import asyncio

from binance import BinanceSocketManager
from binance.client import Client

api_key = "dxFzobejrXqyKYRjE3SB341m7w4oj18Y8c1hl9RRbibT0YCSPaOV6Y1JANnHsQPc"
api_secret = "2BaxckpFPF4uLQ8psXkpukjQA8GPVXbgPvWnO9kDakRosqqLVoVC1oSMjjmGmp2j"
my_account = Client(api_key=api_key, api_secret=api_secret)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
bm = BinanceSocketManager(my_account) # 超时最长时间限制 user_timeout=60)
ts = bm.multiplex_socket(['btcusdt_perpetual@continuousKline_5m', 'btcusdt@kline_5m'])
print(ts)
# TODO:无返回结果，待测试

