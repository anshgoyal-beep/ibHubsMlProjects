import numpy as np
import csv
import sys
import math
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64,skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights
def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))
def replace_null_values_with_mean(X):
    mean = np.nanmean(X,axis = 0)
    indx = np.where(np.isnan(X))
    X[indx] = np.take(mean,indx[1])
    return X

def standardize(X, column_indices):
    means = np.mean(X, axis=0)
    standard_deviations = np.std(X, axis=0)
    for i in column_indices:
        X[:, i] = (X[:, i] - means[i]) / standard_deviations[i]
    return X
def convert_to_numerical_labels(X):
    uniques = np.sort(np.unique(X))
    codes = np.zeros((len(X),), dtype=int)
    for i in range(len(X)):
        for j in range(len(uniques)):
            if X[i] == uniques[j]:
                codes[i] = j
                break
    return codes
def apply_one_hot_encoding(X):
    uniques=np.sort(np.unique(X))
    codes=np.zeros((len(X),len(uniques)),dtype=int)
    for i in range(len(X)):
        for j in range(len(uniques)):
            if X[i]==uniques[j]:
                codes[i,j]=1
                break
    return codes
def convert_given_cols_to_one_hot(X, column_indices):
    X_new=np.zeros((len(X),1),dtype=np.int16)
    for i in range(len(X[0])):
        if i in column_indices:
            codes=apply_one_hot_encoding(X[:,i])
            codes=codes.astype(int)
            X_new=np.append(X_new,codes,axis=1)
        else:
            features=X[:,i].reshape(len(X),1)
            X_new=np.append(X_new,features,axis=1)
    X_new=np.delete(X_new,0,axis=1)
    return X_new


def predict_target_values(test_X, weights):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    test_X=replace_null_values_with_mean(test_X)
    test_X=standardize(test_X,[2,5])
    test_X=convert_given_cols_to_one_hot(test_X,[0,6])
    test_X=np.insert(test_X,0,1.0,axis=1)
    Z=np.dot(test_X,weights)
    A=sigmoid(Z)
    class_predictions = np.where(A >= 0.5, 1, 0)
    return class_predictions

    """
    Note:
    The preprocessing techniques which are used on the train data, should also be applied on the test 
    1. The feature scaling technique used on the training data should be applied as it is (with same mean/standard_deviation/min/max) on the test data as well.
    2. The one-hot encoding mapping applied on the train data should also be applied on test data during prediction.
    3. During training, you have to write any such values (mentioned in above points) to a file, so that they can be used for prediction.
     
    You can load the weights/parameters and the above mentioned preprocessing parameters, by reading them from a csv file which is present in the SubmissionCode.zip
    """
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in regularization.zip and imported properly.
    """

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv")
