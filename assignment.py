#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:23:29 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from helperfuncs import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

class Assignment:

    def __init__(self, filename, delim):
        """
            Initializer for the Assignment class.
            Obtains the data from a given file and shuffles that data.
            Input: 
                filename - name of the file to obtain the data from
                delim - delimiter of the values in that file
        """
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
        

    def logistic_reg(self, folds = 5):
        """
            Implement the training of the logistic regression algorithm (to obtain the best C value) 
            and estimation of the test error after training the algorithm with the full training set
        """
        self.process_data()
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])
        kfold = StratifiedKFold(n_splits = folds)
        
        #Train the algorithm
        c = 1 # we start with c = 1 and we double it every iteration
        c_cross_error_list = []  #To be converted into a matrix to be plotted with the c value, the training error and the validation error
        for iteration in range(1, 21):
            total_train_error = total_val_error = 0
            for train_idx, valid_idx in kfold.split(y_train, y_train):
                train_error, valid_error = calculate_error(4, x_train, y_train, train_idx, valid_idx, c) #Calculate the cross-validation error with the current c
                total_train_error += train_error
                total_val_error += valid_error
                
            c_cross_error_list.append((c, total_train_error, total_val_error))
            c = c * 2
        
        c_cross_error_matrix = np.array(c_cross_error_list) # Convert error list into matrix form
        
        plot_crossVal_err(c_cross_error_matrix) # Plot training and validation errors
        
        #find the best C value
        index_line_of_best_c = np.argmin(c_cross_error_matrix[:, 2])
        best_c = c_cross_error_matrix[index_line_of_best_c, 0]
        
        #Obtain the test error for the best c value
        test_error = calculate_test_error(4, x_train, y_train, x_test, y_test, best_c)
        
        return best_c, test_error #return the best c value and the test error
        
    
    def knn(self):
        pass
    
    
    def bayes(self):
        pass