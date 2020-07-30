import numpy as np
import csv
import sys

from validate import validate

#weights = np.array([-0.4,0.2, -0.4, 0.1, 0.3])
#num_of_iterations = 10
#learning_rate = 0.0001


def import_data(test_X_file_path,test_Y_file_path):
    train_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(test_Y_file_path, delimiter=',', dtype=np.float64)
    return train_X, train_Y

def compute_cost(train_X, train_Y, weights):
    pred_Y = np.dot(train_X, weights)
    mse = np.sum(np.square(pred_Y - train_Y))
    cost_value = mse/(2*len(train_X))
    return cost_value

def compute_gradient_of_cost_function(train_X, train_Y, weights):
    Y_pred = np.dot(train_X, weights)
    diff = Y_pred - train_Y
    dw = (1/(len(train_X)))*np.dot(diff.T, train_X)
    dw = dw.T
    return dw

def change_weights_using_gradient(train_X, train_Y, weights, learning_rate):
    prev_cost = 0
    num_iter = 0
    while True:
        num_iter = num_iter + 1
        dw = compute_gradient_of_cost_function(train_X, train_Y, weights)
        weights = weights - (dw*learning_rate)
        cost = compute_cost(train_X, train_Y, weights)
        #print(prev_cost - cost)
            
        if abs(prev_cost - cost) <= 0.000004:
            print(num_iter, cost)
            break

        if num_iter%1000000 == 0:
            print("i")
        prev_cost = cost
        
    return weights

def train_model(train_X, train_Y):
    train_X = np.insert(train_X, 0, 1, axis=1)
    train_Y = train_Y.reshape(len(train_X), 1)
    weights = np.zeros((train_X.shape[1], 1))
    #dw = compute_gradient_of_cost_function(train_X, train_Y, weights)
    weights = change_weights_using_gradient(train_X, train_Y, weights, 0.00021)
    
    return weights

def save_model(weights, weights_file_name):
    #pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(weights_file_name, 'w', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()
    

if __name__ == "__main__":
    test_X_file_path = "train_X_lr.csv"
    test_Y_file_path = "train_Y_lr.csv"
    train_X, train_Y = import_data(test_X_file_path,test_Y_file_path)
    weights = train_model(train_X, train_Y)
    
    #print(weights)
    save_model(weights, "weights_file.csv")
