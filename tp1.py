#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:51:38 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from helperfuncs import *
from sklearn.utils import shuffle

class Assignment:
    
    def __init__(self, filename, delim):
        self.filename = filename
        self.data = read_data_file(filename, delim)
        self.data = shuffle(self.data)


    def process_data(self, proc_type):
        if (proc_type == "standardize"):
            self.data = standardize_data(self.data[:,:-1], 0)
        else:
            self.data = normalize_data(self.data[:,:-1], 0)
        

    def logist_reg(self):
        pass
    
    def knn(self):
        pass
    
    def bayes(self):
        pass
    
assignment = Assignment("TP1-data.csv", ",")
print(assignment.data)
