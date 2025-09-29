import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from data_cleaning import price_outliers,brand_transformation
data_raw = pd.read_csv('../data/processed/base_model_data.csv')
#to avoid data leakage transform the brands variable in this file
def transform_brands(dataset,brands,global_mean):
    dataset['Brand'] = dataset['Brand'].map(brands)
    dataset['Brand'] = dataset['Brand'].fillna(global_mean)
    dataset['Brand'] = dataset['Brand'].astype('float64')
    return dataset 
#create a train test split
X = data_raw.drop(columns=['Price'])
y = data_raw['Price'].copy()
def train_test_split(X,y,size,state):
    N = len(X)
    n_test = int(N*size)
    indx = np.arange(N)
    np.random.seed(state)
    np.random.shuffle(indx)
    test_indx = indx[:n_test]
    train_indx = indx[n_test:]
    X_train = X.iloc[train_indx].copy()
    X_test = X.iloc[test_indx].copy()
    y_train = y.iloc[train_indx].copy()
    y_test = y.iloc[test_indx].copy()
    return X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2,19)
#temporarily add the y variable to x to transform the brand column
X_train['Price'] = y_train.copy()
brands,global_mean = brand_transformation(X_train,500)
X_train = price_outliers(X_train)
X_train = transform_brands(X_train,brands,global_mean)
#delete the y variable from train set
X_train = X_train.drop(columns=['Price']).copy()
#transform the brand column without data leakage
X_test = transform_brands(X_test,brands,global_mean)
#implement linear regression with gradient descent and MSE
def predict(X,m,b):
    return np.dot(X,m)+b
def MSE(y,y_pred):
    try:
        diff = y - y_pred
        diff_sqr = diff**2
        return np.mean(diff_sqr)
    except OverflowError:
        print(diff,y,y_pred)
        print("ERROR")
        sys.exit(1)
def compute_loss(X,y,m,b):
    y_pred = predict(X,m,b)
    error = y_pred - y
    dm = (np.dot(np.transpose(X),error) * 2)/len(y)
    db = (error * 2)
    return dm,np.mean(db)
def gradient_descent(X,y,batch):
    X = np.array(X,dtype='float')
    y = np.array(y,dtype='float')
    a = 0.002
    m = np.zeros(len((X[0])))
    b = 0
    iter = len(X)/batch
    y_pred = []
    for i in range(int(iter)):
        curr_X = X[batch*i:batch*(i+1)]
        curr_y = y[batch*i:batch*(i+1)]
        y_pred = predict(curr_X,m,b)
        dm,db = compute_loss(curr_X,curr_y,m,b)
        m = m - a*dm
        b = b - a*db
        if i % 10 == 0:
            print(f'MSE: {MSE(curr_y,y_pred)}')
    return m,b
#test the already trained model
def test_model(X,y,m,b):
    X = np.array(X,dtype='float')
    y = np.array(y,dtype='float')
    predictions = predict(X,m,b)
    plt.figure()
    y,y_pred = np.exp(y),np.exp(predictions)
    print(MSE(y,y_pred))
    #print(np.isnan(predictions).any())   # True if any NaNs exist
    #print(np.isinf(predictions).any())   # True if any +/- inf exist
    y_plot = np.clip(y,0,5e5)
    y_pred_plot = np.clip(y_pred,0,5e5)
    plt.hexbin(y_plot,y_pred_plot,gridsize=32)
    plt.show()
    return y_pred,y
m,b = gradient_descent(X_train,y_train,256)
y_pred,y = test_model(X_test,y_test,m,b)
s = 0
s2 = 0
m = 0
for i in range(len(y_pred)):
    d =  abs(y_pred[i] - y[i])
    s += d
    s2 += y[i]
    if d > m:
        m = d
print('Median Error vs Median Price: ', np.median(np.abs(y-y_pred)),np.median(y))
print('Mean Error: ',np.mean(np.abs(y - y_pred)))
print('R2: ',1 - np.sum((y - y_pred)**2) / (np.sum((y - np.mean(y))**2)))
mask = 1e6 > y
print('Mean Error for cheaper cars: ',np.mean(np.abs(y[mask] - y_pred[mask])))
print('R2 for cheaper cars: ',1 - np.sum((y[mask] - y_pred[mask])**2) / (np.sum((y[mask] - np.mean(y[mask]))**2)))
print('Median error vs Median Price for cheaper cars: ',np.median(np.abs(y[mask]-y_pred[mask])),np.median(y[mask]))
