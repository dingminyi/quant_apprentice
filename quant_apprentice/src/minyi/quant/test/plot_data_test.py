# -*- coding: utf-8 -*-
import json
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import Imputer

def load_data(file_name):
    # loading data
    orderbook_json_file = file_name
    fp = open(orderbook_json_file, 'r')
    json_string = fp.readlines()[0]
    fp.close()
    orderbook_json_dict = json.loads(json_string.decode('unicode_escape'))
    data_list = orderbook_json_dict["data"]
    data_list = data_list[300:-1200]
    morning_endtime = datetime.datetime.strptime("11:30:00", "%H:%M:%S").time()
    noon_starttime = datetime.datetime.strptime("1:00:00", "%H:%M:%S").time()
    new_data_list = []
    for data in data_list:
        data_time = datetime.datetime.strptime(data["dataTime"], "%H:%M:%S").time()
        if data_time <  morning_endtime and data_time > noon_starttime:
            new_data_list.append(data)
    data_list = new_data_list
    return DataFrame(data_list)

dataSet1 = load_data(r"D:\Python_Workspace\Pycharm_workspace\Uqer\002230.XSHE2016-08-02-20-47-34.json")
dataSet2 = load_data(r"D:\Python_Workspace\Pycharm_workspace\Uqer\002230.XSHE2016-08-03-16-26-29.json")
dataSet3 = load_data(r"D:\Python_Workspace\Pycharm_workspace\Uqer\002230.XSHE2016-08-04-16-14-37.json")
dataSet4 = load_data(r"D:\Python_Workspace\Pycharm_workspace\Uqer\002230.XSHE2016-08-05-22-32-27.json")
dataSet = pd.concat([dataSet1, dataSet2], axis=0)
dataSet = pd.concat([dataSet, dataSet3], axis=0)
dataSet = pd.concat([dataSet, dataSet4], axis=0)


######################################### Feature Fetching ###########################################

# Features representation

# Basic Set
# V1: price and volume (10 levels)
featV1 = dataSet[['askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 'askVolume1', 'askVolume2',
                  'askVolume3', 'askVolume4', 'askVolume5', 'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4',
                  'bidPrice5', 'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']]
featV1 = np.array(featV1)

# Time-insensitive Set
# V2: bid-ask spread and mid-prices
temp1 = featV1[:, 0:5] - featV1[:, 10:15]
temp2 = (featV1[:, 0:5] + featV1[:, 10:15]) * 0.5
featV2 = np.zeros([temp1.shape[0], temp1.shape[1] + temp2.shape[1]])
featV2[:, 0:temp1.shape[1]] = temp1
featV2[:, temp1.shape[1]:] = temp2

testMidPrice = featV2[:, 5]

plt.figure(figsize=(16,5))
plt.plot(range(featV2.shape[0]),testMidPrice,'b-')
plt.grid()
plt.xlabel('time')
plt.ylabel('midPrice')
plt.show()
