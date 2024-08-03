"""
automated market maker
"""

import base64
import json
import time

import pandas as pd
import requests
from binance.spot import Spot


class AutomatedMarketMaker:
    def __init__(self, trading_pair, api_key, api_secret, base_quote_ratio, spread):
        self.trading_pair = trading_pair  # 交易对，如BTC/USDT
        self.api_key = api_key  # API 密钥
        self.api_secret = api_secret
        self.base_quote_ratio = base_quote_ratio  # 用于生成初始报价的基准比率
        self.spread = spread  # 买卖价差
        self.current_orders = []  # 跟踪当前的订单

    # 1. 数据获取系统
    # 获取订单薄数据
    def fetch_order_book(self, limit):
        """
        :return:
        {"bids":[price,quantity],"asks":[price,quantity]}
        """
        # 获取订单薄数据
        client = Spot(self.api_key, self.api_secret)
        order_book = client.depth(self.trading_pair, limit=limit)  # 默认读取100条
        #        order_book = pd.DataFrame(order_book)
        return order_book

    # 订单薄数据保存
    def save_order_book(self, order_book):
        data = {
            "timestamp": float(order_book['lastUpdateId'][0]),
            "price": {},
            "qty": {}
        }
        for i in range(10):
            data["price"]["bids_" + str(10 - i)] = float(order_book['bids'][10 - i][0])
            data["qty"]["bids_" + str(10 - i)] = float(order_book['bids'][10 - i][1])
        for i in range(10):
            data["price"]["asks_" + str(i + 1)] = float(order_book['asks'][i][0])
            data["qty"]["asks_" + str(i + 1)] = float(order_book['asks'][i][1])
        with open("order_book.json", 'w') as file:
            json.dump(data, file, indent=4)
        return data

    # 订单薄数据处理，分析出过去[interval]时间内新下单的数量、价值、深度信息，以及完成对深度下单的预估
    def process_order_book(self):
        orderbook = self.fetch_order_book(limit=20)
        data = self.save_order_book(orderbook)
        list(data['qty'].values())
        return data

    def get_recent_trades(self, limit):
        # 获取最近成交信息
        client = Spot(self.api_key, self.api_secret)
        # todo:用aggrtades和trades的区别
        trades = client.trades(self.trading_pair, limit=limit)
        # trades = client.agg_trades(self.trading_pair, limit=limit)
        return trades

    def get_market_voltality(self):
        # 获取市场波动数据 - BTC为主, 滚动窗口价格变动
        # TODO：可省
        client = Spot(self.api_key, self.api_secret)
        price_change_BTC = client.rolling_window_ticker("BTCUSDT", windowSize="5m")
        price_change = client.rolling_window_ticker(self.trading_pair, windowSize="5m")
        market_voltality = {"BTC": float(price_change_BTC['priceChangePercent']),
                            "symbol": float(price_change['priceChangePercent'])}
        return market_voltality

    def generate_quote(self, order_book):
        # 基于订单簿生成买入和卖出报价
        best_bid = float(order_book['bids'][0][0])  # 最高买价
        best_ask = float(order_book['asks'][0][0])  # 最低卖价，对应当前价格

        # 在最佳买入和卖出价格中间加上一个固定的spread
        mid_price = (best_bid + best_ask) / 2

        buy_price = mid_price * (1 - self.spread / 2)
        sell_price = mid_price * (1 + self.spread / 2)

        # 调整报价，使其在订单簿的价格范围内
        buy_price = min(buy_price, best_ask)
        sell_price = max(sell_price, best_bid)

        return buy_price, sell_price

    def place_order(self, side, price, size):
        # 向交易所提交订单
        params = {
            "symbol": self.trading_pair,
            "side": side,  # 'buy' or 'sell'
            "type": "LIMIT_MAKER",  # 订单类型, 做市商仅做maker。 limit_maker，立即匹配成为吃单方将被拒绝
            "price": price,
            "quantity": size
        }
        # 添加签名
        params = self._signature(params)
        # 发送请求
        response = requests.post(
            'https://api.binance.com/api/v3/order',
            headers={"X-MBX-APIKEY": self.api_key},
            data=params
        )
        if response.status_code == 200:
            order_id = response.json().get('order_id')
            return order_id
        else:
            raise Exception(f"Failed to place order:{response.status_code}")

    def cancel_order(self, order_id):
        # 取消指定的订单
        params = {
            "symbol": self.trading_pair,
            "orderId": order_id  # 只取消某个订单
        }
        # 添加签名
        params = self._signature(params)
        response = requests.post(
            'https://api.binance.com/api/v3/order',
            headers={"X-MBX-APIKEY": self.api_key},
            data=params
        )
        if response.status_code == 200:
            self.current_orders.remove(order_id)
            return True
        else:
            return False

    def manage_quotes(self, limit):
        # 管理自动报价系统的核心逻辑
        order_book = self.fetch_order_book(limit)
        buy_price, sell_price = self.generate_quote(order_book)

        print(f"Generated Buy Price: {buy_price}, Sell Price: {sell_price}")

        # todo:算法分析订单薄，提供流动性需要对侧怎么下单？

        # 假设我们想要在每一侧下 1 ETH 的订单
        # todo:这个要计算市场的交易热度，看订单薄里散户的单子是什么样的，然后我们下和散户差不多的单来提供
        size = 1

        # 取消当前的挂单
        for order_id in list(self.current_orders):
            self.cancel_order(order_id)

        # 创建新的买卖订单
        buy_order_id = self.place_order('buy', buy_price, size)
        sell_order_id = self.place_order('sell', sell_price, size)

        print(f"Placed Buy Order: {buy_order_id}, Sell Order: {sell_order_id}")

        # 等待一定时间后重新调整
        time.sleep(10)  # 假设每 10 秒重新调整一次

    def _signature(self, params):
        # 参数里加时间戳
        timestamp = int(time.time() * 1000)  # 以毫秒为单位的unix时间戳
        params['timestamp'] = timestamp
        # 加载private_key TODO: 这里考虑是否要加密
        with open('./configs/login.json', 'r') as config_file:
            config = json.load(config_file)
        private_key = config['API_SECRET']
        # 参数中加签名
        payload = '&'.join([f'{param}={value}' for param, value in params.items()])
        signature = base64.b64encode(private_key.sign(payload.encode('ASCII')))
        params['signature'] = signature
        return params


with open('./configs/login.json', 'r') as config_file:
    config = json.load(config_file)
api_key = config['API_KEY']
api_secret = config['API_SECRET']


# 每隔5秒获取一次orderbook

def large_order_check(amount_thread, trading_pair):
    orderbook_dict = {}
    amm = AutomatedMarketMaker(trading_pair, api_key, api_secret, base_quote_ratio=0.5, spread=0.01)
    orderbook_dict['spot'] = amm.fetch_order_book(limit=5000)

    # 市场上挂卖单的情况
    for type, orderbook_ in orderbook_dict.items():
        df = pd.DataFrame(orderbook_['asks'], columns=['price', 'qty'])
        df = df.astype('float64')

        df1 = df[df['qty'] > amount_thread]
        print(trading_pair + type + "市场上挂卖单的情况")
        print(df1.sort_values('qty'))
        # 市场上挂买单的情况
        df = pd.DataFrame(orderbook_['bids'], columns=['price', 'qty'])
        df = df.astype('float64')
        df1 = df[df['qty'] > amount_thread]
        print(trading_pair + type + "市场上挂买单的情况")
        print(df1.sort_values('qty'))
        print("\n")

# 检查大户挂单，阈值、标的symbol
#large_order_check(amount_thread=50000, trading_pair='ZROUSDT')
#large_order_check(amount_thread=10000, trading_pair='WLDUSDT')
large_order_check(amount_thread=10, trading_pair='BTCUSDT')

large_order_check(amount_thread=10000, trading_pair='ZROUSDT')
