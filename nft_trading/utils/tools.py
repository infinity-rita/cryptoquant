import time
def get_timestamp(trade_time):
    # trade_time ='2023-12-26 12:09:20'
    return int(time.mktime(time.strptime(trade_time,"%Y-%m-%d %H:%M:%S")))