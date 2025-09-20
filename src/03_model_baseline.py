import pandas as pd
import numpy as np
import sys
data_raw = pd.read_csv('../data/processed/base_model_data.csv')
def bools(data):
    for column in data:
        data[column] = data[column].apply(lambda x: True if x == "True" else False)
    return data
#data_raw = bools(data_raw)
#print(data_raw.shape)

X = data_raw.drop(columns=['Price'])
print(X.head())
y = data_raw['Price']
def train_test_split(X,y,size,state):
    N = len(X)
    n_test = int(N*size)
    print(n_test)
    indx = np.arange(N)
    print(indx)
    np.random.seed(state)
    np.random.shuffle(indx)
    test_indx = indx[:n_test]
    train_indx = indx[n_test:]
    X_train = X.iloc[train_indx]
    X_test = X.iloc[test_indx]
    y_train = y.iloc[train_indx]
    y_test = y.iloc[test_indx]
    return X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2,11)

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
    db = (error * 2)/len(y)
    return dm,db
def gradient_descent(X,y,batch):
    X = np.array(X,dtype='float')
    y = np.array(y,dtype='float')
    a = 0.0000008
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
    
gradient_descent(X_train,y_train,256)


