"""
# 1.10 TODO：1.优化策略的判断条件 2.私募 1.11 ok，10u调试接口(强平价格部分修改;强平信号到来前后修改)
# 根据V10的版本进行修改

1.11 进入测试阶段
1) 查一下maker和taker两者触发的条件各是什么情况
2) 检查挂单价格的时间先后问题，数据来临的先后问题，和ask_price设置的先后问题
3) 关于实时强制平仓和止盈止损的设置？关于提前获取return? test v2
a.止盈止损？
b.实时强制平仓用：市价吃单
c.提前获取return:15min(14min);10min(7min)

v1:5m快速测试 -- 1小时内是否有误差，anyway，报一下实时获取数据的时间,timestamp

存在的问题：
1.海外代理节点连接不稳，存在反复重连的情况  try - except -- 改用10min测试 - 10/7
2.程序需要部署在云服务器上一直运行，选择香港节点

3.保证金率不充足，可以减少一半本金，2x杠杆，有一定的保证金率

updated by lzy 1.20.2023
1.增加janestreet hf预测收益率
1)加入过去1000时刻的历史数据
"""

import time

import numpy as np
import pandas as pd
from binance.client import Client
# 补充janestreet代码如下：
"""
jane street prediction代码
"""
import warnings

warnings.filterwarnings('ignore')
import gc
import pandas as pd
import numpy as np

import tensorflow as tf

tf.random.set_seed(42)
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

TEST = False

def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2 ** (n + 1 - j)))
    return np.average(a, weights=w)


from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _deprecate_positional_args


class GroupTimeSeriesSplit(_BaseKFold):
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]


import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


# TODO:这两个参数需要调整
n_splits = 2
group_gap = 0  # 可调整，用来调整训练的te # 原训练集用的31


# TODO:training

def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls=1e-2, lr=1e-2):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)

    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)
    out_ae = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='ae_action')(x_ae)

    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)

    out = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='action')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={'decoder': tf.keras.losses.MeanSquaredError(),
                        'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing=ls),
                        'action': tf.keras.losses.BinaryCrossentropy(label_smoothing=ls),
                        },
                  metrics={'decoder': tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                           'ae_action': tf.keras.metrics.AUC(name='AUC'),
                           'action': tf.keras.metrics.AUC(name='AUC'),
                           },
                  )

    return model


# TODO:调整参数

def run(data):
    res = []  # 从1000之后开始做预测
    # for i in range(len(data) - 1000):
    # train = data[i:i + 1000]  # nrows指定读取的行数
    train = data
    features = ['close', 'open']

    print('Filling...')
    # train = train.query('date > 85').reset_index(drop = True)
    # train = train.query('weight > 0').reset_index(drop = True)
    # train[features] = train[features].fillna(method = 'ffill').fillna(0)
    # train['action'] = ((train['resp_1'] > 0) & (train['resp_2'] > 0) & (train['resp_3'] > 0) & (train['resp_4'] > 0) & (train['resp'] > 0)).astype('int')
    train['action'] = (train['close'] < train['close'].shift(-1)).astype('int')  # 作为预测收益率是否为负 True False =1,下一时刻收益率大于零
    resp_cols = ['close', 'open']  # , 'resp_2', 'resp_3', 'resp_4']

    X = train[features][:-1].values
    # y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
    y = train['action'][:-1].reset_index(drop=True)
    date = train['StartTime'][:-1].values
    # weight = train['weight'].values
    # resp = train['resp'].values
    sw = np.mean(np.abs(train[resp_cols][:-1].values), axis=1)
    scores = []
    batch_size = 4096
    gkf = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=group_gap)
    params = {'num_columns': len(features),
              'num_labels': 1,
              'hidden_units': [96, 96, 896, 394, 256],  # , 256],
              'dropout_rates': [0.3527936123679956, 0.38424974585075086, 0.42409238408801436, 0.30431484318345882,
                                0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448],
              'ls': 0,
              'lr': 1e-3,
              }
    for fold, (tr, te) in enumerate(
            gkf.split(train['action'][:-1].values, train['action'][:-1].values, train['StartTime'][:-1].values)):
        ckp_path = f'JSModel_{fold}.hdf5'
        if fold == 1:
            model = create_ae_mlp(**params)
            # checkpoint
            # filepath = "improvement-weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
            filepath = f"fold{fold}_weights-best.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_action_AUC', verbose=1, save_best_only=True, mode='max')
            # ckp = ModelCheckpoint(ckp_path, monitor='val_action_AUC', verbose=0,
            #                      save_best_only=True, save_weights_only=True, mode='max')
            es = EarlyStopping(monitor='val_action_AUC', min_delta=1e-4, patience=10, mode='max',
                               baseline=None, restore_best_weights=True, verbose=0)
            history = model.fit(X[tr], y[tr], validation_data=(X[te], y[te]),
                                sample_weight=sw[tr],
                                epochs=100, batch_size=batch_size, callbacks=[checkpoint, es], verbose=0)
            pre_df = model.predict(X[te])[2]
            th = 0.5
            pre_df = np.where(pre_df >= th, 1, 0).astype(int)
            res.append(pre_df[-1])
            hist = pd.DataFrame(history.history)
            score = hist['val_action_AUC'].max()
            print(f'Fold {fold} ROC AUC:\t', score)
            scores.append(score)
            K.clear_session()
            del model
            rubbish = gc.collect()
            #res1 = pd.DataFrame(res)
    return pre_df[-1]

def timetodate(timestamp):
    timeArray = time.localtime(timestamp / 1000)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


class RealtimeStgTest(object):
    def __init__(self):
        self.signal_record = []  # 记录每一时刻的信号
        self.trading_result = {}  # 策略运行结果记录
        self.ask_price = 0.0  # 最开始不竞价
        self.position = 0  # 最开始无持仓
        try:
            self.bfa = self.connect
            self.bfa.futures_ping()
        except:
            time.sleep(5)
            self.bfa = self.connect
            self.bfa.futures_ping()

    @property
    def connect(self):
        # 定时15min重连，确保和服务器保持连接
        api_key = "dxFzobejrXqyKYRjE3SB341m7w4oj18Y8c1hl9RRbibT0YCSPaOV6Y1JANnHsQPc"
        api_secret = "2BaxckpFPF4uLQ8psXkpukjQA8GPVXbgPvWnO9kDakRosqqLVoVC1oSMjjmGmp2j"
        bfa = Client(api_key=api_key, api_secret=api_secret)
        return bfa

    def get(self):
        "获取计算需要的数据，断连则重连"
        try:
            self.bfa.futures_ping()
        except:
            self.bfa = self.connect
        self.his_data = self.get_historical_kline_data(symbol="ETHUSDT", interval="15m",
                                                       start_time=round((time.time() - 15 * 60 * 4 - 1) * 1000))
        # 用来跑hf_janestreet的历史数据，1000条
        self._run_hf_pred = self.get_historical_kline_data(symbol="ETHUSDT", interval="15m",
                                                           start_time=round((time.time() - 15 * 60 * 1000 - 1) * 1000))
        self._list = [self.his_data['return'].iloc[-3], self.his_data['return'].iloc[-2],
                      self.his_data['return'].iloc[-1]]  # 策略滚动测试部分

    def get_historical_kline_data(self, symbol, interval, start_time):
        '获取历史时刻kline数据,limit 3 计算临近2time return'
        # 香港ip连
        data = self.bfa.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_time)
        data = np.array(data)[:, :8]
        data_list = pd.DataFrame(data,
                                 columns=['StartTime', 'open', 'high', 'low', 'close', 'volume', 'EndTime', 'Amount'])
        data_list = data_list.astype('float64')
        data_list['StartTime'] = data_list['StartTime'].apply(timetodate)
        data_list['EndTime'] = data_list['EndTime'].apply(timetodate)
        data_list['return'] = np.log(data_list['open'] / data_list['open'].shift(1))
        data_list['return'] = data_list['return'].fillna(0)
        return data_list

    def place_maker_order(self, side):
        '做maker,建仓/平仓,limit下单,以最优价格成交'
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

    def liqudation_V1(self, side):
        '做taker,强平/maker,止盈止损单'
        if side == 'BUY':
            self.ask_price = self.his_data['low'].iloc[-1]
        if side == 'SELL':
            self.ask_price = self.his_data['high'].iloc[-1]
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="STOP_MARKET",
                                                             priceProtect="TRUE",
                                                             closePosition="TRUE", workingType='CONTRACT_PRICE',
                                                             stopPrice=self.ask_price, newOrderRespType="RESULT")

    # V2 市价强平止盈止损
    def liqudation(self, side):
        '做taker,强平'
        self.place_order_res = self.bfa.futures_create_order(symbol="ETHUSDT", side=side, type="MARKET",
                                                             quantity=abs(self.positionAmt), newOrderRespType="RESULT")

    def timesync(self):
        '时间同步 -- 建仓'
        u_time = timetodate(time.time() * 1000)[-8:].replace(':', '')
        p_time = self.his_data['StartTime'].iloc[-1][-8:].replace(':', '')
        interval_min = int(p_time[:4]) + 15 - int(u_time[:4]) - 1
        interval_second = 60 - int(u_time[-2:])
        if interval_min >= 0:
            time.sleep(60 * interval_min + interval_second + 1)

    def process(self):
        while True:
            print("策略运行时间:%s" % timetodate(time.time() * 1000))
            self.positionAmt = float(self.bfa.futures_position_information(symbol='ETHUSDT')[0]['positionAmt'])
            if self.positionAmt > 0:
                self.position = 1
            elif self.positionAmt < 0:
                self.position = -1
            else:
                self.position = 0
            self.tot_value = float(self.bfa.futures_account()['totalWalletBalance'])  # 合约账户usdt余额
            print("获取的账户余额数据:%s" % self.tot_value)
            print("持仓情况:%s" % self.position)
            if self.position == 0:
                if self.tot_value != 0:
                    self.get()  # 获取最新价格
                    print("获取最新价格/历史数据的结果如下：")
                    print(self.his_data[['StartTime', 'open', 'high', 'low', 'EndTime']])
                    if self.StrongUpSignal_V3:
                        self.signal_record.append('SU')
                        self.place_maker_order(side='BUY')
                        print("触发强上涨条件，策略做多，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做多挂单未成交")
                        else:
                            print("做多挂单成交")
                        continue
                    if self.WeakUpSignal_V3:
                        self.signal_record.append("WU")
                        self.place_maker_order(side='BUY')
                        print("触发弱上涨条件，策略做多，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做多挂单未成交")
                        else:
                            print("做多挂单成交")
                        continue
                    if self.StrongDownSignal:
                        self.signal_record.append("SD")
                        self.place_maker_order(side='SELL')
                        print("触发强下跌条件，策略做空，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做空挂单未成交")
                        else:
                            print("做空挂单成交")
                        continue
                    if self.WeakDownSignal:
                        self.signal_record.append("WD")
                        self.place_maker_order(side='SELL')
                        print("触发弱下跌条件，策略做空，开单价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            print("做空挂单未成交")
                        else:
                            print("做空挂单成交")
                        continue
            else:
                if self.position == 1:
                    self.get()
                    print("获取最新价格/历史数据的结果如下：")
                    print(self.his_data[['StartTime', 'open', 'high', 'low', 'EndTime']])
                    if self.LongClosed_V2:
                        self.signal_record.append("LC")
                        self.place_maker_order(side='SELL')
                        print("触发平多条件，策略平多，平仓价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            self.liqudation(side='SELL')
                            print("平多挂单未成交,强制平仓")
                        else:
                            print("平多挂单成交!")
                        continue
                if self.position == -1:
                    self.get()
                    print("获取最新价格/历史数据的结果如下：")
                    print(self.his_data[['StartTime', 'open', 'high', 'low', 'EndTime']])
                    if self.ShortClosed_V2:
                        self.signal_record.append("SC")
                        self.place_maker_order(side='BUY')
                        print("触发平空条件，策略平空，平仓价格:%s" % self.ask_price)
                        print("时间同步，等待约15min检验成交")
                        self.timesync()
                        if not self.trade:
                            self.liqudation(side='BUY')
                            print("平空挂单未成交,强制平仓")
                        else:
                            print("平空挂单成交!")
                        continue
            print("================上述条件不符合，等待信号出现，检查接口连接情况，重新获取数据================")
            print("进行时间同步......")
            self.timesync()

    @property
    def LongClosed_V2(self):
        '平多,加入jane street hf prediction'
        self.return_recv = run(self._run_hf_pred)
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SU', 'WU']:
                if self.return_recv <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
                    self.ask_price = self.his_data['high'].iloc[-2]
                    return True
        if self.return_recv <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
            self.ask_price = self.his_data['high'].iloc[-2]
            return True
        return False

    @property
    def ShortClosed_V2(self):
        '平空,加入jane street hf prediction'
        self.return_recv = run(self._run_hf_pred)
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SD', 'WD']:
                if self.return_recv > 0:
                    self.ask_price = self.his_data['low'].iloc[-2]
                    return True
        if self.return_recv > 0:
            self.ask_price = self.his_data['low'].iloc[-2]
            return True
        return False

    @property
    def LongClosed(self):
        '平多'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SU', 'WU']:
                if self._list[2] <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
                    self.ask_price = self.his_data['high'].iloc[2]
                    return True
        if self._list[2] <= 0:  # 这里的计算可以略早于inteval结束,15min, 13min test, 重新取数，
            self.ask_price = self.his_data['high'].iloc[2]
            return True
        return False

    @property
    def ShortClosed(self):
        '平空'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] in ['SD', 'WD']:
                if self._list[2] > 0:
                    self.ask_price = self.his_data['low'].iloc[2]
                    return True
        if self._list[2] > 0:
            self.ask_price = self.his_data['low'].iloc[2]
            return True
        return False

    @property
    def StrongUpSignal_V3(self):
        "强上涨信号"
        if len(self.signal_record) != 0:  # 第一次测试 xx,非第一次不用测试这个
            if self.signal_record[-1] == 'SC' and self._list[1] >= 0 and self._list[2] > self._list[1]:
                self.ask_price = self.his_data['low'].iloc[-2]
                return True
            if self._list[1] >= 0 and self._list[2] > self._list[1]:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        if self._list[1] >= 0 and self._list[2] > self._list[1]:
            self.ask_price = self.his_data['open'].iloc[-1]
            return True
        return False

    @property
    def WeakUpSignal_V3(self):
        "弱上涨信号"
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'SC' and self._list[1] >= 0 and self._list[2] > 0:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        return False

    @property
    def StrongDownSignal(self):
        "强下跌信号"
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and self._list[1] < 0 and self._list[2] < self._list[1]:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
            if self._list[0] < 0 and self._list[1] < self._list[0] and self._list[2] < self._list[1]:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        if self._list[0] < 0 and self._list[1] < self._list[0] and self._list[2] < self._list[1]:
            self.ask_price = self.his_data['open'].iloc[-1]
            return True
        return False

    @property
    def WeakDownSignal(self):
        '弱下跌信号'
        if len(self.signal_record) != 0:
            if self.signal_record[-1] == 'LC' and self._list[1] < 0 and self._list[
                2] < 0:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
            if self._list[0] < 0 and self._list[1] < 0 and self._list[2] < 0:
                self.ask_price = self.his_data['open'].iloc[-1]
                return True
        if self._list[0] < 0 and self._list[1] < 0 and self._list[2] < 0:
            self.ask_price = self.his_data['open'].iloc[-1]
            return True


if __name__ == "__main__":
    r_test = RealtimeStgTest()
    r_test.process()
