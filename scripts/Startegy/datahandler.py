import pandas as pd

secu_data_daily = pd.read_csv("BTCUSDT-5m-2024-07-01.csv", header=None,
                              names=['TimeStamp', "Open", 'High', 'Low', 'Close', 'Volume', "xx", "Amount", "count",
                                     "change1", "chang2", "chang3"])
list_num = [x + 1 for x in range(1, 21)]
for i in list_num:
    if i < 10:
        file_name = "BTCUSDT-5m-2024-07-0" + str(i) + ".csv"
    else:
        file_name = "BTCUSDT-5m-2024-07-" + str(i) + ".csv"
    print(file_name)
    secu_data_monthly_i = pd.read_csv(file_name, header=None,
                                      names=['TimeStamp', "Open", 'High', 'Low', 'Close', 'Volume', "xx", "Amount",
                                             "count",
                                             "change1", "chang2", "chang3"])
    secu_data_daily = pd.concat([secu_data_daily, secu_data_monthly_i])

secu_data_daily.to_csv("2024-BTC-07-5m.csv")
