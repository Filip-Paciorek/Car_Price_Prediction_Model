import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from data_cleaning import price_outliers,brand_transformation
data_raw = pd.read_csv('../data/processed/base_model_data.csv')
#data_raw = bools(data_raw)
#print(data_raw.shape)
def transform_brands(dataset,brands,global_mean):
    dataset['Brand'] = dataset['Brand'].map(brands)
    dataset['Brand'] = dataset['Brand'].fillna(global_mean)
    dataset['Brand'] = dataset['Brand'].astype('float64')
    return dataset 

X = data_raw.drop(columns=['Price'])
y = data_raw['Price'].copy()
def train_test_split(X,y,size,state):
    N = len(X)
    n_test = int(N*size)
    #print(n_test)
    indx = np.arange(N)
    #print(indx)
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
X_train['Price'] = y_train.copy()
brands,global_mean = brand_transformation(X_train,500)
X_train = price_outliers(X_train)
X_train = transform_brands(X_train,brands,global_mean)
X_train = X_train.drop(columns=['Price']).copy()
X_test = transform_brands(X_test,brands,global_mean)
def predict(X,m,b):
    return np.dot(X,m)+b
def MSE(y,y_pred):
    try:
        diff = y - y_pred
        #print(diff)
        #print(y.shape)
        #print(y_pred.shape)
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
    #print(X.shape)
    #print(y.shape)
    m = np.zeros(len((X[0])))
    #print(m.shape)
    b = 0
    iter = len(X)/batch
    y_pred = []
    for i in range(int(iter)):
        curr_X = X[batch*i:batch*(i+1)]
        curr_y = y[batch*i:batch*(i+1)]
        y_pred = predict(curr_X,m,b)
        #print(y_pred.shape)
        dm,db = compute_loss(curr_X,curr_y,m,b)
        m = m - a*dm
        b = b - a*db
        if i % 10 == 0:
            print(f'MSE: {MSE(curr_y,y_pred)}')# dm,db: {dm,db}')
    return m,b
def test_model(X,y,m,b):
    X = np.array(X,dtype='float')
    y = np.array(y,dtype='float')
    #print(X.shape)
    #print(m.shape)
    #print(b.shape)
    predictions = predict(X,m,b)
    plt.figure()
    y,y_pred = np.exp(y),np.exp(predictions)
    print(MSE(y,y_pred))
    #print(np.isnan(predictions).any())   # True if any NaNs exist
    #print(np.isinf(predictions).any())   # True if any +/- inf exist
    indx = []
    for i in range(len(y)):
        indx.append(i)
    y_plot = np.clip(y,0,5e5)
    y_pred_plot = np.clip(y_pred,0,5e5)
    plt.hexbin(y_plot,y_pred_plot,gridsize=32)
    plt.show()
    return y_pred,y
m,b = gradient_descent(X_train,y_train,256)
print(X_test.head())
print(X_train.head())
y_pred,y = test_model(X_test[:500],y_test[:500],m,b)
s = 0
s2 = 0
m = 0
for i in range(len(y_pred)):
    d =  abs(y_pred[i] - y[i])
    s += d
    s2 += y[i]
    if d > m:
        m = d
    #if i > len(y_pred) - 10:
        #print(y[i],y_pred[i], y_pred[i] - y[i])
print(np.median(np.abs(y-y_pred)),np.median(y))
print(np.mean(np.abs(y - y_pred)))
print(1 - np.sum((y - y_pred)**2) / (np.sum((y - np.mean(y))**2)))
mask = 1e6 > y
print(np.mean(np.abs(y[mask] - y_pred[mask])))
print(1 - np.sum((y[mask] - y_pred[mask])**2) / (np.sum((y[mask] - np.mean(y[mask]))**2)))
