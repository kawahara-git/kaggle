#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 16:47:38 2019

@author: kawaharashunsuke
"""

import pandas as pd

class Data:
    def __init__(self,path):
        self.path = path
        
    def read_data(self):
        data = pd.read_csv(self.path)
        return data
    
    def write_data(self,data):
        data.to_csv(self.path,index=False)
        
