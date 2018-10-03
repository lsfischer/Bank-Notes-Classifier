#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:51:38 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from helperfuncs import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

class Assignment:
    def __init__(self, filename, delim):
        self.filename = filename
        self.data = read_data_file(filename, delim)
        self.data = shuffle(self.data)


    def process_data(self, proc_type = "standardize"):
        """
            Processes the data according to the given type of processing requested
        """
        if (proc_type == "standardize"):
            self.data[:, :-1] = standardize_data(self.data[:,:-1], 0)
        else:
            self.data[:, :-1] = normalize_data(self.data[:,:-1], 0)
        

    def logistic_reg(self, folds):
        self.process_data()
        x_training, x_test, y_training, y_test = train_test_split(self.data, self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])
        kfold = StratifiedKFold(n_splits = folds)
        
    
    def knn(self):
        pass
    
    def bayes(self):
        pass
    
assignment = Assignment("TP1-data.csv", ",")
print(assignment.logistic_reg(10))
