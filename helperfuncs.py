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
        plt.xlabel('k')
        
    plt.ylabel('error')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close



def get_prior_and_kdes(x, y, bw):
    class_1 = x[ y[:] == 1, :] #obtain every row that has the last column = 1
    class_0 = x[ y[:] == 0, :] #obtain every row that has the last column = 0

    prior_1 = np.log(len(class_1) / len(x))
    prior_0 = np.log(len(class_0) / len(x))


    feat_kde_score = []   #List that will contain all the different KDE, one for each feature, for all classes


    #Iterate through the features
    for feat in range (0, 4):
        feature_class1 = class_1[:,  [feat]]    #Get a specific feature of the set of class 1
        feature_class0 = class_0[:, [feat]] #Get a specific feature of the set of class 0

        kde_class1 = KernelDensity(kernel = "gaussian", bandwidth = bw)
        kde_class1.fit(feature_class1)
        score_class1 = kde_class1.score(feature_class1)    #is it score, or score_samples

        kde_class0 = KernelDensity(kernel = "gaussian", bandwidth = bw)
        kde_class0.fit(feature_class0)
        score_class0 = kde_class0.score(feature_class0)

        feat_kde_score.append((feat,kde_class1, score_class1))
        feat_kde_score.append((feat, kde_class0, score_class0))

    print(1 - score_class1)
    #print(feat_kde_score)
    
    #Aqui temos que percorrer as features que estão no X
    #Criamos um KDE para cada uma das features para cada uma das classes
    #ou seja vamos ter 4 kde's para para a classe 0 e 4 kde's para a classe 1
    #temos que ser mesmo coluna ou seja, temos que fazer tipo x[:, [1]] em vez de x[:, 1]
    #temos estes kdes todos numa lista de kds
    #nesta função retornamos esta lista de kdes, o prior score da classe 1 e da classe 0
    #noutra função fazemos aqui que o stor escreveu no quadro

    # kde = KernelDensity(kernel = "gaussian", bandwidth = bw)
    # kde.fit(x)
    # kde.score(x)
