import csv
import numpy as np


def import_data():
    X = np.genfromtxt('train_X_lg_v2.csv', delimiter=',', dtype=np.float64, skip_header = 1)
    Y = np.genfromtxt('train_Y_lg_v2.csv', delimiter=',', dtype=np.float64)
    return X,Y


def get_train_data_for_class(X, Y, class_label): 
    class_X = np.copy(X)
    class_Y = np.copy(Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_X, class_Y
    
def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s


def compute_cost(X, Y, W, b):
    m = len(X)
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return cost


def compute_gradient_of_cost_function(X, Y, W, b):
    m = len(X)
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    dW = 1/m * np.dot((A-Y).T,X)
    db = 1/m * np.sum(A-Y)
    dW = dW.T
    return dW, db


def optimize_weights_using_gradient_descent(X,Y,W,b,num_iterations,learning_rate):
    prev_iter_cost = 0
    iter_no = 0
    while True:
        iter_no += 1
        dW, db = compute_gradient_of_cost_function(X, Y, W, b)
        W = W - (learning_rate * dW)
        b = b - (learning_rate * db)
        cost = compute_cost(X,Y,W,b)
        if iter_no % 10000 == 0:
            print(iter_no,cost)
        if abs(cost - prev_iter_cost) < 0.000001:
            print(iter_no,cost)
            break
        prev_iter_cost = cost
    return W, b


def train_model(X, Y):
    Y = Y.reshape(len(X),1)
    W = np.zeros((X.shape[1],1))
    b = 0
    W ,b = optimize_weights_using_gradient_descent(X, Y, W, b, 10, 0.007)
    W = np.insert(W,-1,b)
    return W


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w',newline = '') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()
        
if __name__ == '__main__':
    train_X, train_Y = import_data()
    weights_array = []
    for i in range(4):
        class_label = i
        X, Y = get_train_data_for_class(train_X, train_Y, class_label) 
        weights = train_model(X, Y)
        weights_array.append(weights)
    save_model(weights_array, 'WEIGHTS_FILE.csv')
