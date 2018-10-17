#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:51:38 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from assignment import Assignment   #Import Assignment class from assignment.py

assignment = Assignment("TP1-data.csv")
#assignment.logistic_reg() # Will give a table of information with : Test Error, Precision , Recall, Accuracy and Best C value. It will also plot the error vs Log10 C value
#assignment.knn() # Will give a table of information with : Test Error, Precision , Recall, Accuracy and Best K value. It will also plot the error vs K value
#assignment.bayes() # Will give a table of information with : Test Error, Precision , Recall, Accuracy and Best band-width value. It will also plot the error vs band-width value
assignment.mcNemar_test() #Will compare every classifier using the McNemar test and output the result (takes a while to train all models)

