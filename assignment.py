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
        
        
        

    def logistic_reg(self, folds = 5, x_train = None, x_test = None, y_train = None, y_test = None):
        """
            Implement the training of the logistic regression algorithm (to obtain the best C value) 
            and estimation of the test error after training the algorithm with the full training set
        """
        return self.train_estimate(folds, range(1, 21), "cross_val_err_vs_c.png", "logistic", x_train, x_test, y_train, y_test)
        
    
    def knn(self, folds = 5, x_train = None, x_test = None, y_train = None, y_test = None):
        """
            Implement the training of the KNN algorithm (to obtain the best K value)
            and estimation of the test error after training the algorithm with the full training set
        """
        return self.train_estimate(folds, range(1, 40, 2), "cross_val_err_vs_k.png", "knn", x_train, x_test, y_train, y_test)



    def bayes(self, folds = 5, x_train = None, x_test = None, y_train = None, y_test = None):
        """
            Implement the training of the Naive bayes algorithm (to obtain the best kernel density value) 
            and estimation of the test error after training the algorithm with the full training set
        """
        if(x_train is None):
            self.process_data()
            x_train, x_test, y_train, y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])
        
        cross_error_list = []

        kfold = StratifiedKFold(n_splits = folds)

        for bw in np.arange(0.01, 1.0, 0.02):
            total_train_error = total_val_error = 0

            for train_idx, valid_idx in kfold.split(y_train, y_train):
                x_training_set = x_train[train_idx]
                x_validation_set = x_train[valid_idx]

                y_training_set = y_train[train_idx]
                y_validation_set = y_train[valid_idx]

                prior_kde_list = get_prior_and_kdes(x_training_set, y_training_set, bw)
                train_error, valid_error = calculate_error_bayes(x_training_set, y_training_set, x_validation_set, y_validation_set, prior_kde_list)
                total_train_error += train_error
                total_val_error += valid_error

            cross_error_list.append((bw, total_train_error, total_val_error))

        cross_error_matrix = np.array(cross_error_list)

        
        plot_crossVal_err(cross_error_matrix, "bayes", filename = "cross_val_err_vs_bw.png") # Plot training and validation errors

        index_line_of_best_bw = np.argmin(cross_error_matrix[:, 2])
        best_bw = cross_error_matrix[index_line_of_best_bw, 0]
        
        test_error, predictions = calculate_test_error_bayes(x_test, y_test, x_train, y_train, best_bw)

        print("Best bandwidth: {} \nTest error: {}".format(best_bw, test_error))
        return predictions, test_error

                

    def train_estimate(self, folds, range_to_use, filename, algorithm, x_train, x_test, y_train, y_test):
        """
            Trains the algorithm specified by the "algorithm" paramater. Processes the data, splits it into train and test set,
            performs cross validation to obtain the best value (C value or K value depending on the algorithm) and returns the best value
            and an estimation of the test error
        """

        if(x_train is None):
            self.process_data()
            x_train, x_test, y_train, y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])
        

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

        test_error, predictions = calculate_test_error(4, x_train, y_train, x_test, y_test, int(best_val), algorithm)
        print("Best value: {} \nTest error: {}".format(best_val, test_error))
        return predictions, test_error

    

    def mcNemar_test(self, classifier1, classifier2):
        self.process_data()

        x_train, x_test, y_train, y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])

        if(classifier1 == "knn"):
            predictions1, test_error1 = self.knn(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

        elif(classifier1 == "nb"):
            predictions1, test_error1 = self.bayes(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

        elif(classifier1 == "logistic"):
            predictions1, test_error1 = self.logistic_reg(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

        else:
            print('Choices must be: "knn", "nb", or "logistic"')


        if(classifier2 == "knn"):
            predictions2, test_error2 = self.knn(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

        elif(classifier2 == "nb"):
            predictions2, test_error2 = self.bayes(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

        elif(classifier2 == "logistic"):
            predictions2, test_error2 = self.logistic_reg(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

        else:
            print('Choices must be: "knn", "nb", or "logistic"')

        ground_truth = np.array(y_test)
        predictions1 = np.array(predictions1)
        predictions2 = np.array(predictions2)

        e01 = np.sum(np.logical_and(predictions1 != ground_truth, predictions2 == ground_truth))
        e10 = np.sum(np.logical_and(predictions2 != ground_truth, predictions1 == ground_truth))

        observed_value = ((abs(e01 - e10) - 1) ** 2 )/(e01 + e10)
        critical_point = 3.84 #Chi Squared with 95% confidence and 1 degree of freedom

        if(observed_value >= critical_point):

            if(test_error1 < test_error2):
                print("Classifier {} and {} are significantly different.\nand {} is likely better than {}".format(classifier1, classifier2, classifier1, classifier2)) #They're significantly different, and classifier1 is better than classifier2

            elif(test_error1 > test_error2):
                print("Classifier {} and {} are significantly different.\nand {} is likely better than {}".format(classifier1, classifier2, classifier2, classifier1)) #They're significantly different, and classifier1 is better than classifier2

            else:
                print("Classifier {} and {} are significantly different, but we don't which one is better (their score is the same)".format(classifier1, classifier2)) #They're significantly different but we don't know which one is better

        else:
            print("Classifier {} and {} are not significantly different".format(classifier1, classifier2)) #They're not significantly different