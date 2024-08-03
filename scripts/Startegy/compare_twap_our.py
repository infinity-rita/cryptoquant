import os
import re
import sys
import pdb
import argparse
from time import perf_counter
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from logger import get_logger
from consts import PATHS, TODAY
from utils import parse_date, get_equity_trade_days, remove_number_from_cols, make_parent_dir


def weighted_avg(df, target_col, weight_col):
    df[target_col] = df[target_col] * df[weight_col]
    df[target_col] = df[target_col] / (df[weight_col].sum())
    return df

def long_to_wide(in_df, val_col, suffix_col, uniq_col):
    df = in_df.copy()
    for idx in df.index:
        suffix = df.loc[idx, suffix_col]
        value = df.loc[idx, val_col]
        uniq = df.loc[idx, uniq_col]
        new_col = "_".join([val_col, suffix])
        df.loc[df[uniq_col]==uniq, new_col] = value
    df = df.drop([val_col], axis=1)
    df = df.drop_duplicates()
    return df

def sum_trades(start_date, end_date, process_num,
               in_path, out_path, log_path):
    logger = get_logger(log_file=log_path)
    # make_parent_dir(out_path)
    make_parent_dir(log_path)
    #TODO
    dir_ = Path(in_path).parent
    batch_map = {15:"quarterH", 27:"halfH", 30:"halfH", 57:"1H", 60:"1H", 117:"2H", 120:"2H",}
    # batches = ("0930", "1030", "1300", "1400")

    start_time = perf_counter()
    sub_df = []
    for in_df_path in sorted(dir_.glob("*.csv")):
        sub_df.append(pd.read_csv(in_df_path))
        # batches.append((in_df_path.stem.split("_")[3],in_df_path.stem.split("_")[4]))
    sum_df = pd.concat(sub_df) 
    sum_df = remove_number_from_cols(sum_df)
    sum_df["tradeQty"] = sum_df["tradeQty"].map(abs)
    sum_df = sum_df[sum_df["portfolioid"].notnull()]
    sum_df["date"] = sum_df["StartTime"].str.replace("-","").str[:8]
    sum_df["starttime"] = sum_df["StartTime"].str[11:13] + sum_df["StartTime"].str[14:16]
    sum_df["StartTime"] = pd.to_datetime(sum_df["StartTime"])
    sum_df["EndTime"] = pd.to_datetime(sum_df["EndTime"])
    sum_df["interval"] = ((sum_df["EndTime"] - sum_df["StartTime"])/pd.Timedelta(1,'min')).astype(int) 
    sum_df["batch"] = sum_df["interval"].map(batch_map).fillna("otherH")
    # sum_df["date"] = sum_df["StartTime"].dt.date


    # batches = list(set(batches));pdb.set_trace()

    # for batch in batches:
    for (batch, starttime), dfs in sum_df.groupby(["batch", "starttime"]):
        logger.info(f"{batch} Dealing......")

        # this_in_path = in_path.format(starttime=batch[0], endtime=batch[1])

        # st = datetime.strptime(batch[0], "%H%M")
        # et = datetime.strptime(batch[1], "%H%M")
        # interval = int(str(round((et-st).seconds/60)))
        try:
            if starttime == '0900':
                this_out_path = out_path.format(time=batch,starttime=starttime)  # ;pdb.set_trace()
                this_out_path = os.path.join(os.path.dirname(this_out_path), "0900", os.path.basename(this_out_path))
            else:
                this_out_path = out_path.format(time=batch,starttime=starttime)
        except:
            this_out_path = out_path.format(time="otherH",starttime=starttime)
        make_parent_dir(this_out_path)
        if not os.path.isfile(this_out_path):
            logger.warning(f"[File Created]: {this_out_path}")
            pd.DataFrame(columns=[
                "[1]portfolioid","[2]StartTime","[3]EndTime","[4]BeatTWAP_bps_buy","[5]BeatTWAP_bps_sell"
            ]).to_csv(this_out_path, index=False)

        sum_dfs = []
        datenums = get_equity_trade_days(start_date, end_date)
        for datenum in datenums:
            logger.info ("{:-^70}".format(datenum))#;pdb.set_trace()
            # in_df_path = parse_date(this_in_path, datenum)
            # change_map = {"1457": "1500", "1500": "1457"}
            # lasttime = os.path.basename(in_df_path).split("_")[4]
            # if lasttime in ["1457", "1500"]：
            #     in_df_path2 = os.path.join(os.path.dirname(in_df_path), os.path.basename(in_df_path).replace(lasttime, change_map[lasttime]))
            #     if not os.path.isfile(in_df_path):
            #         logger.error ("[File Not Found]: {}".format(in_df_path))
            #         continue

            # if not os.path.isfile(in_df_path):
            #     logger.error ("[File Not Found]: {}".format(in_df_path))
            #     continue  
            # df = pd.read_csv(in_df_path)
            # df = remove_number_from_cols(df)
            # df = df[df["portfolioid"].notnull()]
            df = dfs[dfs["date"] == datenum].copy()
            # df["portfolioid"] = df["portfolioid"].astype(int)
            df["filledValue"] = df["tradeQty"] * df["tradePrice"]
            df["BeatTWAP_bps"] = df["BeatTWAP_bps"] * df["filledValue"]
            # if datenum == TODAY:
            #     pdb.set_trace()
            df = df.groupby(["StartTime", "side", "portfolioid"]).agg({
                                                        "filledValue": "sum",                        
                                                        "EndTime": "first",
                                                        "BeatTWAP_bps": "sum"}).reset_index()
            df["BeatTWAP_bps"] = df["BeatTWAP_bps"] / df["filledValue"]
            df = df.round({"BeatTWAP_bps": 2})
            df["BeatTWAP_bps_buy"] = 0
            df["BeatTWAP_bps_sell"] = 0
            df["filledValue_buy"] = 0
            df["filledValue_sell"] = 0
            # pdb.set_trace()
            df = long_to_wide(df, "BeatTWAP_bps", "side", "portfolioid")
            df = long_to_wide(df, "filledValue", "side", "portfolioid")
            df = df.drop(["side"], axis=1)

            # pdb.set_trace()

            df = df.groupby("portfolioid").agg({"StartTime": "first",
                                                "filledValue_buy": "first",
                                                "filledValue_sell": "first",
                                                "EndTime": "first",
                                                "BeatTWAP_bps_buy": "first",
                                                "BeatTWAP_bps_sell": "first"}).reset_index()

            # import pdb; pdb.set_trace()
        
            df = df.loc[:, ["portfolioid","StartTime","EndTime","BeatTWAP_bps_buy","BeatTWAP_bps_sell", "filledValue_buy", "filledValue_sell"]]

            df.columns = (f"[{n+1}]{col}" for n, col in enumerate(df.columns))
            sum_dfs.append(df)
            logger.info (f">>> {datenum} Added")

        if sum_dfs:
            tot_df = pd.concat(sum_dfs)
            # pdb.set_trace()
            all_nans = tot_df["[4]BeatTWAP_bps_buy"].isnull() & tot_df["[5]BeatTWAP_bps_sell"].isnull()
            tot_df = tot_df[~all_nans]
            # if os.path.exists(this_out_path):
            #     exist_df = pd.read_csv(this_out_path, dtype={0:str})
            #     tot_df = pd.concat([exist_df, tot_df]).drop_duplicates()
            tot_df.drop_duplicates().to_csv(this_out_path, index=False, sep=",")
        else:
            logger.warning(f"No data added to {batch}!")
    
    logger.info(f"Time used: {round(perf_counter()-start_time, 2)}s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate beat twap by secode.")
    parser.add_argument("start", type=str, help="start date: %Y%m%d")
    parser.add_argument("end", type=str, help="end date: %Y%m%d")
    parser.add_argument("servers", nargs="+", help="the servers to calculate")
    parser.add_argument("-p", "--process_num", type=int, help="process number for calculating twap")
    args = parser.parse_args()

    for srv in args.servers:
        try:
            sum_trades(args.start, args.end, args.process_num,
                    in_path=PATHS["output"][f"{srv}_trades_with_beat_TWAP"],
                    out_path=PATHS["output"][f"{srv}_tot_with_beat_TWAP"],
                    log_path=PATHS["output"][f"{srv}_log"].format("sum_trades"))
        except Exception as e:
            print(e)
