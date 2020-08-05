import numpy as np
import csv
import math

def import_data():
    X = np.genfromtxt("train_X_pr.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_pr.csv", dtype=np.int16, delimiter=',')
    Y = np.reshape(Y, (len(Y), 1))
    return X, Y
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
def get_correlation_matrix(X):
    m=len(X)
    n=len(X[0])
    corr_mat=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            mean_i=np.mean(X[:,i])
            mean_j=np.mean(X[:,j])
            std_dev_i=np.std(X[:,i])
            std_dev_j=np.std(X[:,j])
            numerator=np.sum((X[:,i]-mean_i)*(X[:,j]-mean_j))
            denominator=m*std_dev_i*std_dev_j
            corr=numerator/denominator
            corr_mat[i,j]=corr
            corr_mat[j,i]=corr
    return corr_mat
def select_features(corr_mat, T1, T2):
    n=len(corr_mat)
    filtered_features=[]
    for i in range(1,n):
        if abs(corr_mat[0, i])>T1:
            filtered_features.append(i)
    m=len(filtered_features)
    removed_features=[]
    selected_features=filtered_features.copy()
    for i in range(m):
        for j in range(i+1,m):
            f1=filtered_features[i]
            f2=filtered_features[j]
            if f1 not in removed_features and f2 not in removed_features:
                if abs(corr_mat[f1, f2])>T2:
                    selected_features.remove(f2)
                    removed_features.append(f2)
    return selected_features
def compute_cost(X, Y, W, Lambda):
    m=len(X)
    Z=np.dot(X,W)
    A=sigmoid(Z)
    A[A==1]=0.99999
    A[A==0]=0.00001
    J=-1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))+Lambda/(2*m)*np.sum((W[1:])**2)
    return J
def compute_gradients_using_regularization(X, Y, W, Lambda):
    m=len(X)
    Z=np.dot(X,W)
    A=sigmoid(Z)
    A[A==1]=0.99999
    A[A==0]=0.00001
    dW=Lambda/m*W+1/m*np.dot(X.T,A-Y)
    dW[0]-=Lambda/m*W[0]
    return dW
def optimize_weights_using_gradient_descent(X,Y,W,learning_rate,Lambda):
    previous_iter_cost=0
    iter_number=0
    while True:
        iter_number+=1
        dW=compute_gradients_using_regularization(X,Y,W,Lambda)
        W-=learning_rate*dW
        cost=compute_cost(X,Y,W,Lambda)
        if abs(previous_iter_cost - cost)<0.0000000001:
            print("Breaking at iteration: "+str(iter_number)+" Cost: "+str(cost))
            break
        if iter_number%1000==0:
            print("Iteration number: "+str(iter_number)+" Cost: "+str(cost))
        previous_iter_cost=cost
    return W
def train_model(X,Y):
    W=np.zeros((X.shape[1],1))
    W=optimize_weights_using_gradient_descent(X,Y,W,0.0003,0.49)
    return W
def write_to_csv_file(pred_Y, predicted_Y_file_name):

    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
X,Y=import_data()
X=replace_null_values_with_mean(X)
X=standardize(X, [2, 5])
X=convert_given_cols_to_one_hot(X, [0, 6])
X=np.insert(X, 0, 1.0, axis=1)

W=train_model(X, Y)
write_to_csv_file(W,"WEIGHTS_FILE.csv")
