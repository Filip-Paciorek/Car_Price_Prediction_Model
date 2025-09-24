import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_clean = pd.read_csv('../data/processed/base_model_data.csv')
print(data_clean.head())
def mean_corr(x,y):
    y = np.exp(y)
    l = np.arange(len(x))
    plt.figure()
    plt.scatter(x,y)
    plt.show()
mean_corr(data_clean['Number of Doors'],data_clean['Price'])


