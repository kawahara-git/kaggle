# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../')
from data import *
from preprocessing import * 
from machine_learning import *

def main():
#0.パス指定
    TRAIN_PATH = '../../data/train.csv'

#1.データ取得
    data = Data(TRAIN_PATH)
    train = data.read_data()

#2.前処理
#2-1.カテゴリーデータ処理
    #Sex
    preprocessing = Preprocessing(train)
    train = preprocessing.replace_num(['male','female'],[0,1])
    #Embarked
    preprocessing = Preprocessing(train)
    train = preprocessing.replace_num(['C','Q','S'],[0,1,2])

#2-2.欠損値処理
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna(2)

#2-3.特徴量の変換、追加
    
#2-4.次元削除
    delete_features = ['Name','Cabin','Ticket']
    preprocessing = Preprocessing(train)
    train = preprocessing.delete_features(delete_features)
    
#3.機械学習
#3-0.初期設定
    #検証回数＝100
    trial_n = 100
    result = []
    for i in range(trial_n):

#3-1.訓練データ作成
        machinelearning = MachineLearning(train)
        (train_x,train_y) = machinelearning.create_train_data('Survived')

#3-2.データ分割(学習データ:0.7,検証データ:0.3)
        (train_x,test_x,train_y,test_y) = train_test_split(train_x,train_y,test_size=0.3,random_state=i)

#3-3.学習
        model = RFC(max_depth=30,n_estimators=30,random_state=1)
        model.fit(train_x,train_y)

#3-4.推論
        pred_y = model.predict(test_x)
#3-5.検証
        accuracy_random_forest = accuracy_score(test_y,pred_y)
        #print('Accuracy: {}'.format(accuracy_random_forest))
        
        result.append(accuracy_random_forest)
        print(i)
    avg = sum(result) / trial_n
    print(avg)
        
    print('finish')
    
if __name__ == "__main__":
    main()
