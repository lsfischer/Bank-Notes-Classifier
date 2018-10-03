#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:51:38 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from helperfuncs import *

class Assignment:
    
    def __init__(self, filename, delim):
        self.filename = filename
        self.data = read_data_file(filename, delim)
        
    def logistReg():
        pass
    
    def knn():
        pass
    
    def bayes():
        pass
    
assignment = Assignment("TP1-data.csv", ",")
print(assignment.data)