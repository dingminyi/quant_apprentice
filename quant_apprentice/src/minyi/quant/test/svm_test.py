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

dataSet1 = load_data(r"..\..\..\..\..\002230.XSHE2016-08-02-20-47-34.json")
dataSet2 = load_data(r"..\..\..\..\..\002230.XSHE2016-08-03-16-26-29.json")
dataSet3 = load_data(r"..\..\..\..\..\002230.XSHE2016-08-04-16-14-37.json")
dataSet4 = load_data(r"..\..\..\..\..\002230.XSHE2016-08-05-22-32-27.json")
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

# V3: price differences
temp1 = featV1[:, 4] - featV1[:, 0]
temp2 = featV1[:, 10] - featV1[:, 14]
temp3 = abs(featV1[:, 1:5] - featV1[:, 0:4])
temp4 = abs(featV1[:, 11:15] - featV1[:, 10:14])
featV3 = np.zeros([temp1.shape[0], 1+1+temp3.shape[1]+temp4.shape[1]])
featV3[:, 0] = temp1
featV3[:, 1] = temp2
featV3[:, 2:2+temp3.shape[1]] = temp3
featV3[:, 2+temp3.shape[1]:] = temp4

# V4: mean prices and volumns
temp1 = np.mean(featV1[:, 0:5], 1)
temp2 = np.mean(featV1[:, 10:15], 1)
temp3 = np.mean(featV1[:, 5:10], 1)
temp4 = np.mean(featV1[:, 15:], 1)
featV4 = np.zeros([temp1.shape[0], 1+1+1+1])
featV4[:, 0] = temp1
featV4[:, 1] = temp2
featV4[:, 2] = temp3
featV4[:, 3] = temp4

# V5: accumulated differences
temp1 = np.sum(featV2[:, 0:5], 1)
temp2 = np.sum(featV1[:, 5:10] - featV1[:, 15:], 1)
featV5 = np.zeros([temp1.shape[0], 1+1])
featV5[:, 0] = temp1
featV5[:, 1] = temp2

# Time-insensitive Set
# V6: price and volume derivatives
temp1 = featV1[1:, 0:5] - featV1[:-1, 0:5]
temp2 = featV1[1:, 10:15] - featV1[:-1, 10:15]
temp3 = featV1[1:, 5:10] - featV1[:-1, 5:10]
temp4 = featV1[1:, 15:] - featV1[:-1, 15:]
featV6 = np.zeros([temp1.shape[0] + 1, temp1.shape[1]+temp2.shape[1]+temp3.shape[1]+temp4.shape[1]])
featV6[1:, 0:temp1.shape[1]] = temp1
featV6[1:, temp1.shape[1]: temp1.shape[1]+temp2.shape[1]] = temp2
featV6[1:, temp1.shape[1]+temp2.shape[1]: temp1.shape[1]+temp2.shape[1]+temp3.shape[1]] = temp3
featV6[1:, temp1.shape[1]+temp2.shape[1]+temp3.shape[1]:] = temp4

# combining the features
feat = np.zeros([featV1.shape[0],
                 sum([featV1.shape[1], featV2.shape[1], featV3.shape[1],
                      featV4.shape[1], featV5.shape[1], featV6.shape[1]])
                 ])
feat[:, :featV1.shape[1]] = featV1
feat[:, featV1.shape[1]:featV1.shape[1]+featV2.shape[1]] = featV2
feat[:, featV1.shape[1]+featV2.shape[1]:featV1.shape[1]+featV2.shape[1]+featV3.shape[1]] = featV3
feat[:, featV1.shape[1]+featV2.shape[1]+featV3.shape[1]:featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]] = featV4
feat[:, featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]:featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]+featV5.shape[1]] = featV5
feat[:, featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]+featV5.shape[1]:] = featV6

# normalizing the feature
numFeat = feat.shape[1]
meanFeat = feat.mean(axis=1)
meanFeat.shape = [meanFeat.shape[0], 1]
stdFeat = feat.std(axis=1)
stdFeat.shape = [stdFeat.shape[0], 1]
normFeat = (feat - meanFeat.repeat(numFeat, axis=1))/stdFeat.repeat(numFeat, axis=1)
print "Data Shape: ", normFeat.shape
# print(normFeat)
######################################### Feature Fetching ###########################################

######################################### Data Tagging ###########################################
# mid-price trend of dataset: upward(0),downward(1) or stationary(2)
# 5th column of V2: mid price
upY = featV2[1:, 5] > featV2[:-1, 5]
upY = np.append(upY, 0)
numUp = sum(upY)
print "Num Up: ", numUp
downY = featV2[1:, 5] < featV2[:-1, 5]
downY = np.append(downY, 0)
numDown = sum(downY)
print "Num Down: ", numDown
statY = featV2[1:, 5] == featV2[:-1, 5]
statY = np.append(statY, 0)
numStat = sum(statY)
print "Num Stat: ", numStat

pUp = np.where(upY == 1)[0]  # np.where returns a tuple. pUp are indices where the elements in upY == 1.
pDown = np.where(downY == 1)[0]
pStat = np.where(statY == 1)[0]
multiY = np.zeros([upY.shape[0], 1])
multiY[pUp] = 0
multiY[pDown] = 1
multiY[pStat] = 2

# # divide the dataset into trainSet, and testSst
# numTrain = 1200
# numTest = 500
# # rebalance the ratio of upward, downward and stationary data
# numTrainUp = 250
# numTrainDown = 250
# numTrainStat = 400

# # divide the dataset into trainSet, and testSst
# numTrain = 13000
# numTest = 500
# # rebalance the ratio of upward, downward and stationary data
# numTrainUp = 2000
# numTrainDown = 2000
# numTrainStat = 6000

# divide the dataset into trainSet, and testSst
numTrain = 6500
numTest = 1000
# rebalance the ratio of upward, downward and stationary data
numTrainUp = 1200
numTrainDown = 1200
numTrainStat = 3500

pUpTrain = pUp[:numTrainUp]
pDownTrain = pDown[:numTrainDown]
pStatTrain = pStat[:numTrainStat]

pTrainTemp = np.append(pUpTrain, pDownTrain)
pTrain = np.append(pTrainTemp, pStatTrain)
trainSet = normFeat[pTrain, :]
# trainSet = normFeat[1:numTrain+1,:]
testSet = normFeat[numTrain+1:numTrain+numTest+1, :]

trainMultiYTemp = np.append(multiY[pUpTrain], multiY[pDownTrain])
trainMultiY = np.append(trainMultiYTemp, multiY[pStatTrain])

testMultiY = multiY[numTrain+1:numTrain+numTest+1]
######################################### Data Tagging ###########################################

######################################### SVM Training ###########################################
# training a multi-class svm model
# Model = svm.LinearSVC(C=2.)
Model = svm.SVC(C=10.)
# trainSet = np.nan_to_num(trainSet)
print np.any(np.isnan(trainSet))
print np.all(np.isfinite(trainSet))
print np.any(np.isnan(trainMultiY))
print np.all(np.isfinite(trainMultiY))
trainSet = Imputer().fit_transform(trainSet)
Model.fit(trainSet, trainMultiY)

print np.any(np.isnan(testSet))
print np.all(np.isfinite(testSet))
testSet = Imputer().fit_transform(testSet)
pred = Model.predict(testSet)

ap = Model.score(trainSet, trainMultiY)
print("trainSet mean accuracy: ", ap)

ap = Model.score(testSet, testMultiY)
print("testSet mean accuracy: ", ap)
######################################### SVM Training ###########################################

######################################### Result ###########################################
testMidPrice = featV2[numTrain+1:numTrain+numTest+1, 5]
pUpTest = np.where(pred==0)[0]
pDownTest = np.where(pred==1)[0]
pStatTest = np.where(pred==2)[0]

plt.figure(figsize=(16,5))
plt.plot(range(numTest),testMidPrice,'b-',pUpTest,testMidPrice[pUpTest],'r.',pDownTest,testMidPrice[pDownTest],'g.')
plt.grid()
plt.xlabel('time')
plt.ylabel('midPrice')
plt.show()

######################################### Result ###########################################
