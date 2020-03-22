#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:48:12 2019

@author: kawaharashunsuke
"""

import pandas as pd

class MachineLearning:
    def __init__(self,data):
        self.data = data
        
    def create_train_data(self,y_feature):
        train_x = self.data.drop([y_feature], axis=1)
        train_y = self.data[y_feature]
        return train_x,train_y