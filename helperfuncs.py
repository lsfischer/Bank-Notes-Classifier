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
from sklearn.metrics import accuracy_score

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
    class_1 = x[ y[:] == 1, :] #obtain every row that has the last column = 1
    class_0 = x[ y[:] == 0, :] #obtain every row that has the last column = 0

    prior_1 = np.log(len(class_1) / len(x)) #Obtain the prior probability of class 1
    prior_0 = np.log(len(class_0) / len(x)) #Obtain the prior probability of class 0


    kde_list = []   #List that will contain all the different KDE, one for each feature, for all classes

    #Iterate through the features

    for feat in range (0, 4):
        feature_class1 = class_1[:,  [feat]]    #Get a specific feature of the set of class 1
        feature_class0 = class_0[:, [feat]] #Get a specific feature of the set of class 0

        kde_class1 = KernelDensity(kernel = "gaussian", bandwidth = bw)
        kde_class1.fit(feature_class1)  #Fit the kde of class 1 for feature "feat"

        kde_class0 = KernelDensity(kernel = "gaussian", bandwidth = bw)
        kde_class0.fit(feature_class0) #Fit the kde of class 0 for feature "feat"
        

        kde_list.append((kde_class0, kde_class1))   #In kde_list we store the KDE for feature "feat" for class 0 and for class 1

    return (prior_0, prior_1, kde_list)


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
    
    logs_list_class1_train = []
    logs_list_class0_train = []

    logs_list_class1_valid = []
    logs_list_class0_valid = []
    
    
    for feat in range(0,4):
        feature_column_train = x_training[:, [feat]]
        feature_column_valid = x_validation[:, [feat]]

        kde_to_use_class0 = kde_list[feat][0] #We get the kde for feat and class 0 
        kde_to_use_class1 = kde_list[feat][1] #We get the kde for feat and class 1
        scores_class1_train = kde_to_use_class1.score_samples(feature_column_train)
        scores_class0_train = kde_to_use_class0.score_samples(feature_column_train)

        scores_class1_valid = kde_to_use_class1.score_samples(feature_column_valid)
        scores_class0_valid = kde_to_use_class0.score_samples(feature_column_valid)

        logs_list_class1_train.append(scores_class1_train)
        logs_list_class0_train.append(scores_class0_train)

        logs_list_class1_valid.append(scores_class1_valid)
        logs_list_class0_valid.append(scores_class0_valid)

    logs_matrix_class1_train = np.column_stack((logs_list_class1_train[0], logs_list_class1_train[1], logs_list_class1_train[2], logs_list_class1_train[3]))
    logs_matrix_class0_train = np.column_stack((logs_list_class0_train[0], logs_list_class0_train[1], logs_list_class0_train[2], logs_list_class0_train[3]))

    logs_matrix_class1_valid = np.column_stack((logs_list_class1_valid[0], logs_list_class1_valid[1], logs_list_class1_valid[2], logs_list_class1_valid[3]))
    logs_matrix_class0_valid = np.column_stack((logs_list_class0_valid[0], logs_list_class0_valid[1], logs_list_class0_valid[2], logs_list_class0_valid[3]))

    sum_feat_class1_train =  prior_class1+ np.sum(logs_matrix_class1_train, axis = 1)
    sum_feat_class0_train =  prior_class0 + np.sum(logs_matrix_class0_train, axis = 1)

    sum_feat_class1_valid =  prior_class1+ np.sum(logs_matrix_class1_valid, axis = 1)
    sum_feat_class0_valid =  prior_class0 + np.sum(logs_matrix_class0_valid, axis = 1)

    
    prediction_list_train = (sum_feat_class1_train >= sum_feat_class0_train).astype(int)
    prediction_list_valid = (sum_feat_class1_valid >= sum_feat_class0_valid).astype(int)

    return ((1 - accuracy_score(y_training, prediction_list_train), 1 - accuracy_score(y_validation, prediction_list_valid)))

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

    prior_kde_list = get_prior_and_kdes(x_full_train, y_full_train, best_bw)

    prior_class0 = prior_kde_list[0]
    prior_class1 = prior_kde_list[1]
    kde_list = prior_kde_list[2]
    
    logs_list_class1_test = []
    logs_list_class0_test = []   
    
    for feat in range(0,4):
        feature_column_test = x_test[:, [feat]]

        kde_to_use_class0 = kde_list[feat][0] #We get the kde for feat and class 0 
        kde_to_use_class1 = kde_list[feat][1] #We get the kde for feat and class 1
        scores_class1_test = kde_to_use_class1.score_samples(feature_column_test)
        scores_class0_test = kde_to_use_class0.score_samples(feature_column_test)

        logs_list_class1_test.append(scores_class1_test)
        logs_list_class0_test.append(scores_class0_test)


    logs_matrix_class1_test = np.column_stack((logs_list_class1_test[0], logs_list_class1_test[1], logs_list_class1_test[2], logs_list_class1_test[3]))
    logs_matrix_class0_test = np.column_stack((logs_list_class0_test[0], logs_list_class0_test[1], logs_list_class0_test[2], logs_list_class0_test[3]))

    sum_feat_class1_test =  prior_class1+ np.sum(logs_matrix_class1_test, axis = 1)
    sum_feat_class0_test =  prior_class0 + np.sum(logs_matrix_class0_test, axis = 1)
    
    prediction_list_test = (sum_feat_class1_test >= sum_feat_class0_test).astype(int)

    return (1 - accuracy_score(y_test, prediction_list_test))