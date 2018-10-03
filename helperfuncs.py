#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:47:38 2018

@author: lucas
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def standardize_data(data, column):
    """
        Standerdize the data from the given column forward
    """
    mean = np.mean(data[:, column:], axis = 0)  #calculate the mean of all the columns from the given column forward
    std = np.std(data[:, column:])  #Calculate the standard deviation for all the columns from the given column forward
    data[:, column:] = (data[:, column:] - mean)/std #standerdize the data
    return data


def normalize_data(data, column):
    """
        Normalize the data from the given column forward
    """
    min_data = np.min(data[:, column:]) #calculate the minimum for all the columns from the given column forward
    max_data = np.max(data[:, column:]) #calculate the maximum for all the columns from the given column forward
    data[:, column:] = (data[:, column:] - min_data) / (max_data - min_data)
    return data
    


def read_data_file(filename, delim):
    """ 
        Reads the data file and gets data separated by delimiter delim 
    """
    """ input: filename to read (filename), delimeter (delim) """
    """ ouput: data """
    # Load the data from file
    data = np.loadtxt(filename,delimiter=delim)
    return data


def calculate_error(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C = C, tol = 1e-10)
    reg.fit(X[train_ix, :feats], Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:, 1]    
    squares = (prob-Y) ** 2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])