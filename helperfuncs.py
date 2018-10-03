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
    # Load the data from file
    data = np.loadtxt(filename,delimiter=delim)
    return data


def calculate_error(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return the cross validation error"""
    reg = LogisticRegression(C = C, tol = 1e-10)
    reg.fit(X[train_ix, :feats], Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:, 1]    
    squares = (prob-Y) ** 2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])


def plot_crossVal_err(err_array, if_log_c_axis = True, filename = 'cross_val_err_vs_c.png'):
    """ 
        Plots training and cross-validation errors vs C parameter
        if if_log_c_axis = true, C axis is displayed in log scale
        err_array[:,0] -> C values
        err_array[:,1] -> training errors
        err_array[:,2] -> validation errors
    """
    plt.figure()
    if (if_log_c_axis):
        plt.plot(np.log10(err_array[:,0]),err_array[:,1],"-r", label="training")    
        plt.plot(np.log10(err_array[:,0]),err_array[:,2],"-b", label="validation")
        plt.xlabel('$\log_{10}(C)$')
    else:
        plt.plot(err_array[:,0],err_array[:,1],"-r", label="training")    
        plt.plot(err_array[:,0],err_array[:,2],"-b", label="validation")
        plt.xlabel('C')
    plt.ylabel('error')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
