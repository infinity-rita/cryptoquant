"""
automated market maker
"""

import base64
import json
import time

import requests


class AutomatedMarketMaker:
    def __init__(self, trading_pair, api_key, base_quote_ratio, spread):
        self.trading_pair = trading_pair  # 交易对，如BTC/USDT
        self.api_key = api_key  # API 密钥
        self.base_quote_ratio = base_quote_ratio  # 用于生成初始报价的基准比率
        self.spread = spread  # 买卖价差
        self.current_orders = []  # 跟踪当前的订单

    def fetch_order_book(self):
        # 获取订单薄数据
        params = {
            "symbol": self.trading_pair,
            "limit": 10  # 买卖10档
        }
        # 添加签名
        params = self._signature(params)
        response = requests.get(
            'https://api.binance.com/api/v3/depth',
            headers={"X-MBX-APIKEY": self.api_key},
            data=params
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch order book:{response.status_code}")

    def generate_quote(self, order_book):
        # 基于订单薄生成买入和卖出报价
        pass

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

    def manage_quotes(self):
        # 管理自动报价系统的核心逻辑
        pass

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

    def connect_test(self):
        #测试服务器连通性
        return requests.get('https://api.binance.com/api/v3/ping')
# 测试代码情况

trading_pair = 'BTCUSDT'
with open('./configs/login.json', 'r') as config_file:
    config = json.load(config_file)
api_key = config['API_KEY']

amm = AutomatedMarketMaker(trading_pair, api_key, base_quote_ratio=0.5, spread=0.01)

amm.fetch_order_book()