# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:50:45 2019
@author: s-moh
"""
import numpy as np
import pandas as pd
import datetime

dataBefore = '../_data/yoochoose-origin/yoochoose-clicks.dat' #Path to Original Training Dataset "Clicks" File
dataTestBefore = '../_data/yoochoose-origin/yoochoose-test.dat' #Path to Original Testing Dataset "Clicks" File
dataAfter = '../_data/yoochoose-prep/' #Path to Processed Dataset Folder
dayTime = 86400 #Validation Only one day = 86400 seconds

session_key='sessionid'
item_key='itemid'
time_key='timestamp'

def create_dirs(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dirs(dataAfter)

def removeShortSessions(data):
    #delete sessions of length < 1
    sessionLen = data.groupby(session_key).size() #group by sessionID and get size of each session
    data = data[np.in1d(data[session_key], sessionLen[sessionLen > 1].index)]
    return data

#Read Dataset in pandas Dataframe (Ignore Category Column)
train = pd.read_csv(dataBefore, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
test = pd.read_csv(dataTestBefore, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
train.columns = [session_key, time_key, item_key] #Headers of dataframe
test.columns = [session_key, time_key, item_key] #Headers of dataframe
train[time_key]= train[time_key].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #Convert time objects to timestamp
test[time_key] = test[time_key].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #Convert time objects to timestamp

#remove sessions of less than 2 interactions
train = removeShortSessions(train)
#delete records of items which appeared less than 5 times
itemLen = train.groupby(item_key).size() #groupby itemID and get size of each item
train = train[np.in1d(train[item_key], itemLen[itemLen > 4].index)]
#remove sessions of less than 2 interactions again
train = removeShortSessions(train)

######################################################################################################3
'''
#Separate Data into Train and Test Splits
timeMax = data[time_key].max() #maximum time in all records
sessionMaxTime = data.groupby(session_key)[time_key].max() #group by sessionID and get the maximum time of each session
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index #training split is all sessions that ended before the last day
sessionTest  = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index #testing split is all sessions has records in the last day
train = data[np.in1d(data[session_key'], sessionTrain)]
test = data[np.in1d(data[session_key'], sessionTest)]
'''
#Delete records in testing split where items are not in training split
test = test[np.in1d(test[item_key], train[item_key])]
#Delete Sessions in testing split which are less than 2
test = removeShortSessions(test)

#Convert To CSV
#print('Full Training Set has', len(train), 'Events, ', train[session_key'].nunique(), 'Sessions, and', train[item_key].nunique(), 'Items\n\n')
#train.to_csv(dataAfter + 'recSys15TrainFull.txt', sep='\t', index=False)
print('Testing Set has', len(test), 'Events, ', test[session_key].nunique(), 'Sessions, and', test[item_key].nunique(), 'Items\n\n')
test.to_csv(dataAfter + 'yoochoose-test.txt', sep=',', index=False)

######################################################################################################3
#Separate Training set into Train and Validation Splits
timeMax = train[time_key].max()
sessionMaxTime = train.groupby(session_key)[time_key].max()
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index #training split is all sessions that ended before the last 2nd day
sessionValid = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index #validation split is all sessions that ended during the last 2nd day
trainTR = train[np.in1d(train[session_key], sessionTrain)]
trainVD = train[np.in1d(train[session_key], sessionValid)]
#Delete records in validation split where items are not in training split
trainVD = trainVD[np.in1d(trainVD[item_key], trainTR[item_key])]
#Delete Sessions in testing split which are less than 2
trainVD = removeShortSessions(trainVD)
#Convert To CSV
print('Training Set has', len(trainTR), 'Events, ', trainTR[session_key].nunique(), 'Sessions, and', trainTR[item_key].nunique(), 'Items\n\n')
trainTR.to_csv(dataAfter + 'yoochoose-train.txt', sep=',', index=False)
print('Validation Set has', len(trainVD), 'Events, ', trainVD[session_key].nunique(), 'Sessions, and', trainVD[item_key].nunique(), 'Items\n\n')
trainVD.to_csv(dataAfter + 'yoochoose-valid.txt', sep=',', index=False)