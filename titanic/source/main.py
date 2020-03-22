# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RFC

from data import *
from preprocessing import *
from machine_learning import *

def main():
#0.パス指定
    TRAIN_PATH = '../data/train.csv'
    TEST_PATH = '../data/test.csv'
    RESULT_PATH = '../data/result.csv'
#1.データ取得
    #訓練データ
    data = Data(TRAIN_PATH)
    train = data.read_data()
    #テストデータ
    data = Data(TEST_PATH)
    test = data.read_data()

#2.前処理
#2-1.カテゴリーデータ処理
#train
    #Sex
    preprocessing = Preprocessing(train)
    train = preprocessing.replace_num(['male','female'],[0,1])
    #Embarked
    preprocessing = Preprocessing(train)
    train = preprocessing.replace_num(['C','Q','S'],[0,1,2])
#test
    #Sex
    preprocessing = Preprocessing(test)
    test = preprocessing.replace_num(['male','female'],[0,1])
    #Embarked
    preprocessing = Preprocessing(test)
    test = preprocessing.replace_num(['C','Q','S'],[0,1,2])
    
#2-2.欠損値処理
#train
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna(2)
#test
    test["Age"] = test["Age"].fillna(train["Age"].median())
    test["Embarked"] = test["Embarked"].fillna(2)
    test["Fare"] = test["Fare"].fillna(train["Fare"].median())

#2-3.特徴量の変換、追加
    
#2-4.次元削除
    delete_features = ['Name','Cabin','Ticket']
#train
    preprocessing = Preprocessing(train)
    train = preprocessing.delete_features(delete_features)
#test
    preprocessing = Preprocessing(test)
    test = preprocessing.delete_features(delete_features)
    
#3.機械学習
#3-1.訓練データ作成
    machinelearning = MachineLearning(train)
    (train_x,train_y) = machinelearning.create_train_data('Survived')
    
#3-2.学習
    model = RFC(max_depth=30,n_estimators=30,random_state=1)
    model.fit(train_x,train_y)
    
#3-3.推論
    pred_y = model.predict(test)
    print(pred_y)
    
#4.データ作成
    #data = Data(RESULT_PATH)
    id = test.loc[:,['PassengerId']]
    #print(id)
    survived = pd.DataFrame({'Survived':pred_y})
    #print(survived)
    result = pd.concat([id,survived], axis = 1)
    #result = result.set_index(['PassengerId', 'Survived'],inplace=True)
    print(result)
    data = Data(RESULT_PATH)
    data.write_data(result)    
    
    print('finish')
    
    
if __name__ == "__main__":
    main()
