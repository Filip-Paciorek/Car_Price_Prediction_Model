import pandas as pd
import numpy as np
import re
data_raw = pd.read_csv('../data/raw/Moroccan Used Cars Dataset (MUCars - 2024)/cars_dataframe.csv')

print(data_raw.shape)
print(data_raw.columns)
def clean(dataset):
    data = dataset.dropna(subset=['Price','Brand','Model','Mileage'])
    #print(data.shape)
    return data
base_data = clean(data_raw)

def base_model_data(dataset):
    data_clean = dataset.dropna()
    #print(data_clean.shape)
    #print(data_clean.head())
    condition_check = data_clean[['Condition','Price']]
    condition_check = condition_check.groupby('Condition')
    (data_clean.loc[data_clean['Year'].str.len() > 4,'Year']) = data_clean.loc[data_clean['Year'].str.len() > 4,'Year'].str.slice(0,4) 
    data_clean = data_clean[data_clean['Price'] <= 1000000]
    #data_clean = data_clean[~((data_clean['Price'] > 1000000) & (data_clean['Year'].astype(int) <2010))].sort_values('Year')
    #print(data_clean)
    data_clean['Price'] = data_clean['Price']/1000
    print(data_clean['Price'].head())
    data_clean['Fiscal Power'] = data_clean['Fiscal Power'].str.replace('CV','')
    data_clean['Fiscal Power'] = data_clean['Fiscal Power'].apply(lambda x: int(x) if x.isdigit() else int("".join(re.findall('\d+',x))))
    #print(data_clean['Fiscal Power'].unique())
    #print(data_clean['Fiscal Power'])
    #data_clean = data_clean.drop(data_clean[data_clean['Price'] > 1000000])
    data_clean['Mileage'] = data_clean['Mileage'].str.replace(' ','').str.split('-')
    #print(data_clean['Mileage'].head())
    data_clean['Mileage'] = data_clean['Mileage'].apply(lambda x: np.log(np.mean([int(i) if i.isdigit() else int("".join(re.findall('\d+', i))) for i in x])+1))
    #data_clean['Mileage'] = data_clean['Mileage'].apply(lambda x: int(x))
    
    print(data_clean['Mileage'].head())
    print(data_clean['Condition'].unique())
    condition = {'New':7,'Excellent':6,'Very Good':5,'Good':4,'Fair':3,'Damaged':2,'For Parts':1}
    data_clean['Condition'] = data_clean['Condition'].map(lambda x: condition[x])
    data_clean['Number of Doors'] = data_clean['Number of Doors'].map(lambda x: int(x))
    data_clean = pd.get_dummies(data_clean,columns=['Gearbox'],drop_first=True)
    data_clean = pd.get_dummies(data_clean,columns=['First Owner'],drop_first=True)
    data_clean = pd.get_dummies(data_clean,columns=['Origin'])
    data_clean = pd.get_dummies(data_clean,columns=['Fuel'])
    min_appearences = 35
    appearances = data_clean['Brand'].value_counts()
    rare_brands = appearances[min_appearences > appearances].index
    data_clean['Brand'] = data_clean['Brand'].replace(rare_brands,'Other')
    data_clean = pd.get_dummies(data_clean,columns=['Brand'])
    data_clean.drop(columns=['Model','Sector','Location','Equipment'],inplace=True)
    return data_clean

    #data_clean.drop(columns=['Equipment','Origin',]
base_model_data = base_model_data(base_data)
base_model_data.to_csv('../data/processed/base_model_data.csv',index=False)
