#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:42:35 2019

@author: kawaharashunsuke
"""

import pandas as pd

class Preprocessing:
    def __init__(self,data):
        self.data = data
        
    def delete_features(self,features):
        data = self.data.drop(columns=features)
        return data
    
    def add_feature(self,feature_name):
        self.data[feature_name]=0
    
    def replace_num(self,word,num):
        data = self.data.replace(word,num)
        return data