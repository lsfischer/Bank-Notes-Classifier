#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:47:38 2018

@author: Lucas Fischer
@author: Joana Martins
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from NaiveBayes import NaiveBayes

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
    data = np.loadtxt(filename,delimiter=delim) # Load the data from file
    return data



def calculate_error(feats, X, Y, train_ix, valid_ix, value, algorithm):
    """return the cross validation error using Logistic regression"""
    reg = LogisticRegression(C = value, tol=1e-10) if(algorithm == "logistic") else KNeighborsClassifier(n_neighbors = value)
    reg.fit(X[train_ix, :feats], Y[train_ix])
    accuracy_training = reg.score(X[train_ix, :feats], Y[train_ix])
    accuracy_validation = reg.score(X[valid_ix, :feats], Y[valid_ix])
    return 1 - accuracy_training, 1 - accuracy_validation



def calculate_test_error(feats, X_train, Y_train, X_test, Y_test, value, algorithm):
    """
        return the test error, training with the full training set, using Logistic regression
    """
    reg = LogisticRegression(C = value, tol=1e-10) if(algorithm == "logistic") else KNeighborsClassifier(n_neighbors = value)
    reg.fit(X_train[:, :feats], Y_train[:])
    return 1 - reg.score(X_test[:, :feats], Y_test[:])



def plot_crossVal_err(err_array, algorithm, if_log_c_axis = True, filename = 'cross_val_err_vs_c.png'):
    """ 
        Plots training and cross-validation errors vs C parameter
        if if_log_c_axis = true, C axis is displayed in log scale
        err_array[:,0] -> C/K values
        err_array[:,1] -> training errors
        err_array[:,2] -> validation errors
    """
    plt.figure()
    if(algorithm == "logistic"):
        if (if_log_c_axis):
            plt.plot(np.log10(err_array[:,0]), err_array[:,1], "-r", label="training")
            plt.plot(np.log10(err_array[:,0]), err_array[:,2], "-b", label="validation")
            plt.xlabel('$\log_{10}(C)$')
        else:
            plt.plot(err_array[:,0], err_array[:,1], "-r", label="training")    
            plt.plot(err_array[:,0], err_array[:,2], "-b", label="validation")
            plt.xlabel('C')
    else:
        plt.plot(err_array[:,0], err_array[:, 1], "-r", label="training")
        plt.plot(err_array[:,0], err_array[:, 2], "-b", label="validation")
        if(algorithm == "knn"):
            plt.xlabel('k')
        else:
            plt.xlabel('bw')
        
    plt.ylabel('error')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close



def get_prior_and_kdes(x, y, bw):
    return NaiveBayes().get_prior_and_kdes(x, y, bw, num_feats = 4)


def calculate_error_bayes(x_training, y_training, x_validation, y_validation, prior_kde_list):
    """
        Calculates the training and validation error of the Naive Bayes classifier

        Parameters:
            x_training : Values of the training set
            y_training : Labels of the training set
            x_validation : Values of the validation set
            y_validation : Labels of the validation set
            prior_kde_list: List containing the prior probability of class 0 and 1, aswell as a list with KDE's for all features (for both classes)

        Returns:
            The training and validation error
    """
    prior_class0 = prior_kde_list[0]
    prior_class1 = prior_kde_list[1]
    kde_list = prior_kde_list[2]
    nb = NaiveBayes()

    return nb.calculate_training_error(x_training, y_training, x_validation, y_validation, prior_class0, prior_class1, kde_list, num_feats = 4)

def calculate_test_error_bayes(x_test, y_test, x_full_train, y_full_train, best_bw):
    """
        Calculates the test error of the Naive Bayes classifier

        Parameters:
            x_test : Values of the test set
            y_test : Labels of the test set
            x_full_train : Values of the full set to use for training
            y_full_train : Labels of the full set to use for training

        Returns:
            The test error for the model training with the full set and the best bw found before
    """

    nb = NaiveBayes()
    prior_kde_list = nb.get_prior_and_kdes(x_full_train, y_full_train, best_bw, num_feats = 4)

    prior_class0 = prior_kde_list[0]
    prior_class1 = prior_kde_list[1]
    kde_list = prior_kde_list[2]
    
    return nb.calculate_test_error(x_test, y_test, x_full_train, y_full_train, best_bw, prior_class0, prior_class1, kde_list, num_feats = 4)