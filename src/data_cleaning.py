import pandas as pd
import numpy as np
import re
data_clean = pd.read_csv('../data/processed/clean_data')
def price_outliers(dataset):
    dataset = dataset[dataset['Price'] <= 1000000].copy()
    return dataset
def brand_transformation(dataset,k):
    #get the mean of all prices for each car
    subset = dataset[['Brand','Price']]
    brands = dataset['Brand'].unique()
    global_mean = np.mean(dataset['Price'])
    sums = {}
    count = {}
    means = {}
    for brand in brands:
        sums[brand] = 0
        count[brand] = 0
        means[brand] = 0
    #if brand in dataset, price into sum, number into count
    for brand in brands:
        for price in subset.query(f"Brand == @brand")['Price']:
            sums[brand] += price
            count[brand] += 1
    for brand in brands:
        means[brand] = (sums[brand] +(k*global_mean))/(count[brand]+k)
    return means,global_mean
def base_model_data(data_clean):
    #convert price to safe log
    data_clean['Price'] = np.log(data_clean['Price'].clip(lower=1))
    #make a new feature to strengthen the model
    data_clean['MoY'] = data_clean['Mileage']/(2025-data_clean['Year'])
    #assign ordinal values to the condition variable
    condition = {'New':7,'Excellent':6,'Very Good':5,'Good':4,'Fair':3,'Damaged':2,'For Parts':1}
    data_clean['Condition'] = data_clean['Condition'].map(lambda x: condition[x])
    #create a new feature to strengthen the model
    data_clean['CtY'] = data_clean['Condition']*data_clean['Year'].apply(lambda x:x/1000)
    #bring all numbers to the same magnitude
    data_clean['Year'] = np.log(data_clean['Year'].clip(lower=1))
    data_clean['Mileage'] = np.log(data_clean['Mileage'].clip(lower=1))
    data_clean['CtY'] = np.log(data_clean['CtY'].clip(lower=1))
    data_clean['MoY'] = np.log(data_clean['MoY'].clip(lower=1))
    #change doors from float to int
    data_clean['Number of Doors'] = data_clean['Number of Doors'].map(lambda x: int(x))
    #onehot encode variables that are not ordinal
    data_clean = pd.get_dummies(data_clean,columns=['Gearbox'],drop_first=True)
    data_clean = pd.get_dummies(data_clean,columns=['First Owner'],drop_first=True)
    data_clean = pd.get_dummies(data_clean,columns=['Origin'])
    data_clean = pd.get_dummies(data_clean,columns=['Fuel'])
    #drop columns not used in the base model due to incompatibility/lack of value
    data_clean.drop(columns=['Mileage','Model','Sector','Location','Equipment'],inplace=True)
    return data_clean
base_model_data = base_model_data(data_clean)
base_model_data.to_csv('../data/processed/base_model_data.csv',index=False)
