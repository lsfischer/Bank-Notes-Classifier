import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity

class NaiveBayes:
    """
        Class responsible for implementing all the Naive Bayes classifier logic
    """

    def __init__(self):
        pass


    def get_prior_and_kdes(self, x, y, bw, num_feats = 4):
        """
            Calculates the prior probability for each class and creates a list filled with KDE fitted for each feature

            Args:
                x         - Matrix with the feature values
                y         - Vector with the target values
                bw        - Band-width value to use
                num_feats - Number of features to use

            Returns:
                the prior probability for each class and creates a list filled with KDE fitted for each feature
        """
        class_1 = x[ y[:] == 1, :] #obtain every row that has the last column = 1
        class_0 = x[ y[:] == 0, :] #obtain every row that has the last column = 0

        prior_1 = np.log(len(class_1) / len(x)) #Obtain the prior probability of class 1
        prior_0 = np.log(len(class_0) / len(x)) #Obtain the prior probability of class 0


        kde_list = []   #List that will contain all the different KDE, one for each feature, for all classes

        #Iterate through the features

        for feat in range (0, num_feats):
            feature_class1 = class_1[:,  [feat]]    #Get a specific feature of the set of class 1
            feature_class0 = class_0[:, [feat]] #Get a specific feature of the set of class 0

            kde_class1 = KernelDensity(kernel = "gaussian", bandwidth = bw)
            kde_class1.fit(feature_class1)  #Fit the kde of class 1 for feature "feat"

            kde_class0 = KernelDensity(kernel = "gaussian", bandwidth = bw)
            kde_class0.fit(feature_class0) #Fit the kde of class 0 for feature "feat"
            

            kde_list.append((kde_class0, kde_class1))   #In kde_list we store the KDE for feature "feat" for class 0 and for class 1

        return (prior_0, prior_1, kde_list)





    def calculate_training_error(self, x_training, y_training, x_validation, y_validation, prior_class0, prior_class1, kde_list, num_feats = 4):
        """
            Calculates the training and validation error of the Naive Bayes classifier

            Args:
                x_training     - Values of the training set
                y_training     - Labels of the training set
                x_validation   - Values of the validation set
                y_validation   - Labels of the validation set
                prior_kde_list - List containing the prior probability of class 0 and 1, aswell as a list with KDE's for all features (for both classes)

            Returns:
                The training and validation error
        """

        logs_list_class1_train = [] #List in wich to store the sub-lists containg the probability logs of a feature given class is 1 (for the trainig set)
        logs_list_class0_train = [] #List in wich to store the sub-lists containg the probability logs of a feature given class is 0 (for the trainig set)

        logs_list_class1_valid = [] #List in wich to store the sub-lists containg the probability logs of a feature given class is 1 (for the validation set)
        logs_list_class0_valid = [] #List in wich to store the sub-lists containg the probability logs of a feature given class is 0 (for the validation set)

        for feat in range(0, num_feats):
            feature_column_train = x_training[:, [feat]]    #Get the column of feature "feat" from the training set
            feature_column_valid = x_validation[:, [feat]]  #Get the column of feature "feat" from the validation set

            kde_to_use_class0 = kde_list[feat][0] #We get the kde fitted for feature "feat" and class 0
            kde_to_use_class1 = kde_list[feat][1] #We get the kde fitted for feature "feat" and class 1

            scores_class1_train = kde_to_use_class1.score_samples(feature_column_train) #Obtain a column of probability logs of feature "feat" given the class is 1, for the training set 
            scores_class0_train = kde_to_use_class0.score_samples(feature_column_train) #Obtain a column of probability logs of feature "feat" given the class is 0, for the training set 

            scores_class1_valid = kde_to_use_class1.score_samples(feature_column_valid) #Obtain a column of probability logs of feature "feat" given the class is 0, for the validation set
            scores_class0_valid = kde_to_use_class0.score_samples(feature_column_valid) #Obtain a column of probability logs of feature "feat" given the class is 0, for the validation set

            #Store the probabilitiy log columns in these lists
            logs_list_class1_train.append(scores_class1_train)  
            logs_list_class0_train.append(scores_class0_train)

            logs_list_class1_valid.append(scores_class1_valid)
            logs_list_class0_valid.append(scores_class0_valid)

        #Create a new matrix composed of 4 columns that correspond to the probability of each feature given some class, for the training and validation set
        logs_matrix_class1_train = np.column_stack((logs_list_class1_train[0], logs_list_class1_train[1], logs_list_class1_train[2], logs_list_class1_train[3]))
        logs_matrix_class0_train = np.column_stack((logs_list_class0_train[0], logs_list_class0_train[1], logs_list_class0_train[2], logs_list_class0_train[3]))

        logs_matrix_class1_valid = np.column_stack((logs_list_class1_valid[0], logs_list_class1_valid[1], logs_list_class1_valid[2], logs_list_class1_valid[3]))
        logs_matrix_class0_valid = np.column_stack((logs_list_class0_valid[0], logs_list_class0_valid[1], logs_list_class0_valid[2], logs_list_class0_valid[3]))

        #For every row, sum every column in that row plus the prior probability of class 1 or 0, for the training and validation set
        sum_feat_class1_train =  prior_class1 + np.sum(logs_matrix_class1_train, axis = 1)
        sum_feat_class0_train =  prior_class0 + np.sum(logs_matrix_class0_train, axis = 1)

        sum_feat_class1_valid =  prior_class1 + np.sum(logs_matrix_class1_valid, axis = 1)
        sum_feat_class0_valid =  prior_class0 + np.sum(logs_matrix_class0_valid, axis = 1)

        #Obtain the prediction list by checking which class maximizes the previous calculation
        prediction_list_train = (sum_feat_class1_train >= sum_feat_class0_train).astype(int)
        prediction_list_valid = (sum_feat_class1_valid >= sum_feat_class0_valid).astype(int)

        return ((1 - accuracy_score(y_training, prediction_list_train), 1 - accuracy_score(y_validation, prediction_list_valid)))





    def calculate_test_error(self, x_test, y_test, x_full_train, y_full_train, best_bw, prior_class0, prior_class1, kde_list, num_feats = 4):
        """
            Calculates the test error of the Naive Bayes classifier

            Args:
                x_test : Values of the test set
                y_test : Labels of the test set
                x_full_train : Values of the full set to use for training
                y_full_train : Labels of the full set to use for training

            Returns:
                The test error for the model and a list with the class predictions, after training with the full set and the best band-with found before
        """
        logs_list_class1_test = [] #List in wich to store the sub-lists containg the probability logs of a feature given class is 1
        logs_list_class0_test = [] #List in wich to store the sub-lists containg the probability logs of a feature given class is 0

        for feat in range(0,num_feats):
            feature_column_test = x_test[:, [feat]]

            kde_to_use_class0 = kde_list[feat][0] #We get the kde fitted for feature "feat" and class 0 
            kde_to_use_class1 = kde_list[feat][1] #We get the kde fitted for feature "feat" and class 1

            scores_class1_test = kde_to_use_class1.score_samples(feature_column_test) #Obtain a column of probability logs of feature "feat" given the class is 1
            scores_class0_test = kde_to_use_class0.score_samples(feature_column_test) #Obtain a column of probability logs of feature "feat" given the class is 0

            #Store the probabilitiy log columns in these lists
            logs_list_class1_test.append(scores_class1_test)
            logs_list_class0_test.append(scores_class0_test)

        #Create a new matrix composed of 4 columns that correspond to the probability of each feature given some class
        logs_matrix_class1_test = np.column_stack((logs_list_class1_test[0], logs_list_class1_test[1], logs_list_class1_test[2], logs_list_class1_test[3]))
        logs_matrix_class0_test = np.column_stack((logs_list_class0_test[0], logs_list_class0_test[1], logs_list_class0_test[2], logs_list_class0_test[3]))

        #For every row, sum every column in that row plus the prior probability of class 1 or 0
        sum_feat_class1_test =  prior_class1 + np.sum(logs_matrix_class1_test, axis = 1)
        sum_feat_class0_test =  prior_class0 + np.sum(logs_matrix_class0_test, axis = 1)
        
        #Obtain the prediction list by checking which class maximizes the previous calculation
        prediction_list_test = (sum_feat_class1_test >= sum_feat_class0_test).astype(int)

        return (1 - accuracy_score(y_test, prediction_list_test), prediction_list_test)