import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
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
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2,198)

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
    print(X.shape)
    print(m.shape)
    print(b.shape)
    predictions = predict(X,m,b)
    print(MSE(y,predictions))
    plt.figure()
    y,predictions = np.exp(y),np.exp(predictions)
    y = y.astype(int)
    
    print(np.isnan(predictions).any())   # True if any NaNs exist
    print(np.isinf(predictions).any())   # True if any +/- inf exist
    y_pred = predictions.astype(int)
    indx = []
    for i in range(len(y)):
        indx.append(i)
    plt.hexbin(y,y_pred,gridsize=32)
    plt.show()
    return y_pred,y
m,b = gradient_descent(X_train,y_train,256)

y_pred,y = test_model(X_test[:100],y_test[:100],m,b)
s = 0
s2 = 0
for i in range(len(y_pred)):
    s += abs(y_pred[i] - y[i])
    s2 += y[i]
    #print(y[i],y_pred[i], y_pred[i] - y[i])
print(s2,s)

