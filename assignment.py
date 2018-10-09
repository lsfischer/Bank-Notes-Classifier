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
        return self.train_estimate(folds, range(1, 21), "cross_val_err_vs_c.png", "logistic")
        
    
    def knn(self, folds = 5):
        """
            Implement the training of the KNN algorithm (to obtain the best K value)
            and estimation of the test error after training the algorithm with the full training set
        """
        return self.train_estimate(folds, range(1, 40, 2), "cross_val_err_vs_k.png", "knn")
        

    def train_estimate(self, folds, range_to_use, filename, algorithm):
        """
            Trains the algorithm specified by the "algorithm" paramater. Processes the data, splits it into train and test set,
            performs cross validation to obtain the best value (C value or K value depending on the algorithm) and returns the best value
            and an estimation of the test error
        """
        self.process_data()
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])

        kfold = StratifiedKFold(n_splits = folds)
        cross_error_list = []  #To be converted into a matrix to be plotted with the c value, the training error and the validation error

        val = 1 #Value to use, could be C value for logistic regression or K value for nearest neighbours
        for k in range_to_use:
            total_train_error = total_val_error = 0
            for train_idx, valid_idx in kfold.split(y_train, y_train):
                train_error, valid_error = calculate_error(4, x_train, y_train, train_idx, valid_idx, val, algorithm) #Calculate the cross-validation error with the current c
                total_train_error += train_error
                total_val_error += valid_error
                
            cross_error_list.append((val, total_train_error, total_val_error))

            if(algorithm == "logistic"):
                val *= 2
            else:
                val = k
        
        cross_error_matrix = np.array(cross_error_list) # Convert error list into matrix form

        plot_crossVal_err(cross_error_matrix, algorithm, filename = filename) # Plot training and validation errors

        #find the best value (C or K)
        index_line_of_best_val = np.argmin(cross_error_matrix[:, 2])
        best_val = cross_error_matrix[index_line_of_best_val, 0]

        test_error = calculate_test_error(4, x_train, y_train, x_test, y_test, int(best_val), algorithm)
        return best_val, test_error


        
    def bayes(self):
        pass