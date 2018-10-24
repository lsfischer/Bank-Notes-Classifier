#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:23:29 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from helper_funcs import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold

class Assignment:
    """
        Class responsible for the entire process of executing a classifier. (Reading the data, processesing it, splitting into different sets, training and estimating test error)

        Attributes:
            filename - Name of the data file to read
            delim    - Value delimiter of the given file
    """

    def __init__(self, filename, delim = ","):
        """
            Initializer for the Assignment class.
            Obtains the data from a given file and shuffles that data.

            Args: 
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
            self.data[:, :-1] = standardize_data(self.data[:, :-1], 0)
        else:
            self.data[:, :-1] = normalize_data(self.data[:, :-1], 0)
        
        
        

    def logistic_reg(self, folds = 5, x_train = None, x_test = None, y_train = None, y_test = None, is_mcnemar_test = False):
        """
            Implement the training of the logistic regression algorithm (to obtain the best C value) 
            and estimation of the test error after training the algorithm with the full training set

            Args:
                folds           - Number of folds to use (default is 5)
                x_train         - Training set of the values
                x_test          - Test set of the values
                y_train         - Training set of the target-values
                y_test          - Test set of the target-values
                is_mcnemar_test - Boolean used to determine wheather or not to plot the graph

            Returns:
                A list of class predictions Logistic Regression predicted after training, and the test error
        """

        return self.train_estimate(folds, range(1, 21), "cross_val_err_vs_c.png", "logistic", x_train, x_test, y_train, y_test, is_mcnemar_test)
        
    
    def knn(self, folds = 5, x_train = None, x_test = None, y_train = None, y_test = None, is_mcnemar_test = False):
        """
            Implement the training of the KNN algorithm (to obtain the best K value)
            and estimation of the test error after training the algorithm with the full training set

            Args:
                folds           - Number of folds to use (default is 5)
                x_train         - Training set of the values
                x_test          - Test set of the values
                y_train         - Training set of the target-values
                y_test          - Test set of the target-values
                is_mcnemar_test - Boolean used to determine wheather or not to plot the graph

            Returns:
                A list of class predictions K-Nearest-Neighbours predicted after training, and the test error
        """

        return self.train_estimate(folds, range(1, 40, 2), "cross_val_err_vs_k.png", "knn", x_train, x_test, y_train, y_test, is_mcnemar_test)



    def bayes(self, folds = 5, x_train = None, x_test = None, y_train = None, y_test = None, is_mcnemar_test = False):
        """
            Implement the training of the Naive bayes algorithm (to obtain the best kernel density value) 
            and estimation of the test error after training the algorithm with the full training set

            Args:
                folds           - Number of folds to use (default is 5)
                x_train         - Training set of the values
                x_test          - Test set of the values
                y_train         - Training set of the target-values
                y_test          - Test set of the target-values
                is_mcnemar_test - Boolean used to determine wheather or not to plot the graph

            Returns:
                A list of class predictions Naive Bayes predicted after training, and the test error 
        """

        #If we provide the method with the training and test sets (from the mcnemar_test method) we don't need to do it again
        if(x_train is None):
            self.process_data()
            x_train, x_test, y_train, y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])
            print("Calculating class predictions and cross-validation errors for all bandwidth values ... This takes a few seconds\n")
        
        cross_error_list = []

        

        kfold = StratifiedKFold(n_splits = folds)

        #For loop iterating every band-width value from 0.01 to 1.0 with a step of 0.02
        for bw in np.arange(0.01, 1.0, 0.02):
            total_train_error = total_val_error = 0

            #Stratified k folds
            for train_idx, valid_idx in kfold.split(y_train, y_train):

                #Obtain the training and validation folds from the training set
                x_training_set = x_train[train_idx]
                x_validation_set = x_train[valid_idx]

                y_training_set = y_train[train_idx]
                y_validation_set = y_train[valid_idx]

                prior_kde_list = get_prior_and_kdes(x_training_set, y_training_set, bw) #Get the prior probability of each class and a list with fitted KDE's for every feature
                train_error, valid_error = calculate_error_bayes(x_training_set, y_training_set, x_validation_set, y_validation_set, prior_kde_list) #Get the cross-validation error
                total_train_error += train_error
                total_val_error += valid_error

            cross_error_list.append((bw, total_train_error, total_val_error))

        cross_error_matrix = np.array(cross_error_list) #Conver the error list into matrix form
        
        #Find the best band-with value
        index_line_of_best_bw = np.argmin(cross_error_matrix[:, 2])
        best_bw = cross_error_matrix[index_line_of_best_bw, 0]
        
        test_error, predictions = calculate_test_error_bayes(x_test, y_test, x_train, y_train, best_bw)

        if(not is_mcnemar_test):
            precision, recall = get_metrics(predictions, y_test) #Get the precision and recall

            print("\nTest error\t|\tPrecision\t|\tRecall\t|\tAccuracy\t|\tBest band-width")
            print("{:.4f}\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\n".format(test_error, precision, recall, (1 - test_error), best_bw))

            plot_crossVal_err(cross_error_matrix, "bayes", filename = "cross_val_err_vs_bw.png") # Plot training and validation errors
        
        return predictions, test_error

                

    def train_estimate(self, folds, range_to_use, filename, algorithm, x_train, x_test, y_train, y_test, is_mcnemar_test):
        """
            Trains the algorithm specified by the "algorithm" paramater. Processes the data, splits it into train and test set,
            performs cross validation to obtain the best value (C value or K value depending on the algorithm) and returns the best value
            and an estimation of the test error

            Args:
                folds           - Number of folds to use (default is 5)
                range_to_use    - range to be used in the for loop
                filename        - The name of the image file to store the plot
                algorithm       - The algorithm to run (logistic - Logistic Regression, knn - K Nearest Neighbours)
                x_train         - Training set of the values
                x_test          - Test set of the values
                y_train         - Training set of the target-values
                y_test          - Test set of the target-values
                is_mcnemar_test - Boolean used to determine wheather or not to plot the graph

            Returns:
                A list of class predictions the classifier predicted after training, and the test error for that classifier
        """

        #If we provide the method with the training and test sets (from the mcnemar_test method) we don't need to do it again
        if(x_train is None):
            self.process_data()
            x_train, x_test, y_train, y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])
        

        kfold = StratifiedKFold(n_splits = folds)
        cross_error_list = []  #To be converted into a matrix to be plotted with the c value, the training error and the validation error

        val = 1 #Value to use, could be C value for logistic regression or K value for nearest neighbours
        for k in range_to_use:
            total_train_error = total_val_error = 0
            
            #Stratified k folds
            for train_idx, valid_idx in kfold.split(y_train, y_train):
                train_error, valid_error = calculate_error(4, x_train, y_train, train_idx, valid_idx, val, algorithm) #Calculate the cross-validation error with the current c or k value
                total_train_error += train_error
                total_val_error += valid_error
                
            cross_error_list.append((val, total_train_error, total_val_error))

            if(algorithm == "logistic"):
                val *= 2    #If we're in logistic regression we want to duplicate the value
            else:
                val = k #If we're in knn we want the value to be equal to k
        
        cross_error_matrix = np.array(cross_error_list) # Convert error list into matrix form
            
        #find the best value (C or K)
        index_line_of_best_val = np.argmin(cross_error_matrix[:, 2])
        best_val = cross_error_matrix[index_line_of_best_val, 0]

        test_error, predictions = calculate_test_error(4, x_train, y_train, x_test, y_test, int(best_val), algorithm)

        if(not is_mcnemar_test):
            if(algorithm == "logistic"):
                value_string = "C"
            else:
                value_string = "K"

            precision, recall = get_metrics(predictions, y_test) #Get the precision and recall

            print("\nTest error\t|\tPrecision\t|\tRecall\t|\tAccuracy\t|\tBest {}".format(value_string))
            print("{:.4f}\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\n".format(test_error, precision, recall, (1 - test_error), best_val))

            plot_crossVal_err(cross_error_matrix, algorithm, filename = filename) # Plot training and validation errors
        
        return predictions, test_error

    

    def mcNemar_test(self):
        """
            Method that calculates the McNemar test for all combination of classifiers.
            This method compares KNN vs Logistic Regression, KNN vs Naive Bayes and Naive Bayes vs Logistic Regression
        """

        self.process_data()

        x_train, x_test, y_train, y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size = 0.33, stratify = self.data[:, -1])

        print("Calculating class predictions and test errors for all classifiers ... This takes a few seconds\n")

        #Obtain the list of class predictions and the test error for all classifiers, using the same data sets
        logistic_prediction, logistic_test_error = self.logistic_reg(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, is_mcnemar_test = True) 
        knn_prediction, knn_test_error = self.knn(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, is_mcnemar_test = True)
        bayes_prediction, bayes_test_error = self.bayes(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, is_mcnemar_test = True)

        #KNN vs Logistic Regression
        calculate_mcnemar(np.array(knn_prediction), np.array(logistic_prediction), np.array(y_test), knn_test_error, logistic_test_error, "K-Nearest-Neighbours", "Logistic Regression")
        print("\n----------------------------------------------------\n")

        #KNN vs Naive Bayes
        calculate_mcnemar(np.array(knn_prediction), np.array(bayes_prediction), np.array(y_test), knn_test_error, bayes_test_error, "K-Nearest-Neighbours", "Naive Bayes")
        print("\n----------------------------------------------------\n")

        #Naive Bayes vs Logistic Regression
        calculate_mcnemar(np.array(bayes_prediction), np.array(logistic_prediction), np.array(y_test), bayes_test_error, logistic_test_error, "Naive Bayes", "Logistic Regression")
        print("\n----------------------------------------------------\n")