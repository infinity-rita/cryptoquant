import requests
import json
import pandas as pd 
import time
import os

import time
def get_timestamp(trade_time):
    # trade_time ='2023-12-26 12:09:20'
    return int(time.mktime(time.strptime(trade_time,"%Y-%m-%d %H:%M:%S")))

def get_tradetime(timestamp):
    return time.strftime("%H:%M:%S",time.localtime(timestamp))

def get_trade_dt(timestamp):
    return time.strftime("%Y-%m-%d",time.localtime(timestamp))

class nftmarketplaceAPI(object):

    def __init__(self,apikey_info): 
        self.opensea_apikey=apikey_info['opensea']
        self.magiceden_apikey=apikey_info['magic_eden']
        self.tensor_apikey=apikey_info['tensor']

    def get_collection_events_opensea(self, params, start_time, end_time):

        url = f"https://api.opensea.io/api/v2/events/collection/{params['collection_slug']}?after={start_time}&before={end_time}&event_type={params['event_type']}"


        headers = {
            "accept": "application/json",
            "x-api-key": self.opensea_apikey
        }

        response = requests.get(url, headers=headers)

        if response.status_code!=200:
            print("opensea api connect failed!")
        
        json_data = json.loads(response.text)['asset_events']
        result=[]

        for tmp in json_data:
            tmp1 = {}
            tmp1['event_type'] = tmp['event_type']
            tmp1['chain'] = tmp['chain']
            tmp1['nft identifier'] = tmp['nft']['identifier']
            tmp1['quantity'] = tmp['quantity']
            tmp1['payment quantity'] = round(int(tmp['payment']['quantity'])/(10**tmp['payment']['decimals']),4) # 暂保留四位小数
            tmp1['payment symbol'] = tmp['payment']['symbol'] # currency, eg: ETH
            tmp1['trade_time'] = get_tradetime(tmp['event_timestamp'])
            tmp1['trade_dt'] = get_trade_dt(tmp['event_timestamp'])
            tmp1['event_timestamp'] = tmp['event_timestamp']
            result.append(tmp1)
        
        with open('./result/test_opensea.json', 'w') as json_file:
            json.dump(result, json_file, indent=4)

        return result

class DataHandler(nftmarketplaceAPI):
    def __init__(self,apikey_info,params):
        super(DataHandler,self).__init__(apikey_info)
        self.params = params
    
    def get_data(self):
        # 获取给定start_time,end_time时间内的数据
        start_time = get_timestamp(self.params['after_timestamp'])
        end_time = get_timestamp(self.params['before_timestamp'])
        end_time_ref = end_time
        new_order = ["trade_dt","trade_time","event_type","chain","nft identifier","quantity","payment quantity","payment symbol","event_timestamp"]

        while end_time <= end_time_ref:
            result = self.get_collection_events_opensea(self.params,start_time,end_time)
            new_df=pd.DataFrame(result).reindex(columns=new_order)
            new_df=new_df.sort_values(by=['event_timestamp'])
            if os.path.exists(self.params['tmp_csv_path']):
                df=pd.read_csv(self.params['tmp_csv_path'])
                df =pd.concat([df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=['trade_dt', 'trade_time'], keep='last')
                df.to_csv(self.params['tmp_csv_path'],index=False)
            else:
                new_df.to_csv(self.params['tmp_csv_path'],index=False)
            end_time = new_df['event_timestamp'].iloc[-1]
            
    def get_klines(self):
        # 分频率撮合 成不同的open,high,low,close,volume,amount,freq,avg_price,count klines数据
        df = pd.read_csv(self.params['tmp_csv_path'])
        
        df['event_timestamp'] = df['event_timestamp'].astype(int)

        df['Time']=df['event_timestamp'].apply(get_tradetime)

        df['Time'] = pd.to_datetime(df['Time'])

        df=df.set_index('Time')

        new_df = df[['quantity','payment quantity']]
        # 修改df含有的列，仅包括quantity和payment quantity
        for freq,csv_path in self.params['klines'].items():
            new_df_kmin=new_df.resample(freq).sum()
            new_df_kmin.to_csv(csv_path)

    def plot(self):
        pass

    def main(self):
        pass

if __name__ == "__main__":
    with open('./configs/api_key.json', 'r') as config_file:
            apikey_info = json.load(config_file)
    with open('./configs/collection_info.json', 'r') as config_file:
            params = json.load(config_file) 

    connect = DataHandler(apikey_info,params)
    # 方法覆盖
    connect.get_klines()