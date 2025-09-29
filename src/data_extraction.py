import pandas as pd
import numpy as np
import re
#read the data in
data_raw = pd.read_csv('../data/raw/Moroccan Used Cars Dataset (MUCars - 2024)/cars_dataframe.csv')
if __name__ == "__main__":
    #bring every brand to lowercase
    data_raw['Brand'] = data_raw['Brand'].str.lower()
    #check for nulls and check shape
    print(data_raw.isnull().sum())
    print(data_raw.shape)
    #drop all missing values
    data_raw = data_raw.dropna().copy()
    #check shape again in order to see how much was dropped
    print(data_raw.shape)
    #check random values for irregularities in each condition 
    condition_check = data_raw[['Condition','Price']]
    condition_check = condition_check.groupby('Condition')
    for i in condition_check:
        print(i)
    #change each string of year which categorises something else other than just the year to the first four digits
    (data_raw.loc[data_raw['Year'].str.len() > 4,'Year']) = data_raw.loc[data_raw['Year'].str.len() > 4,'Year'].str.slice(0,4)     
    #change values with letters to just numbers
    data_raw['Fiscal Power'] = data_raw['Fiscal Power'].str.replace('CV','')
    data_raw['Fiscal Power'] = data_raw['Fiscal Power'].apply(lambda x: int(x) if str(x).isdigit() else int("".join(re.findall('\d+',x))))
    #change ranges of miles to mean
    data_raw['Mileage'] = data_raw['Mileage'].str.replace(' ','').str.split('-')
    data_raw['Mileage'] = data_raw['Mileage'].apply(lambda x:int((np.mean([int(i) if i.isdigit() else int("".join(re.findall('\d+', i))) for i in x])+1)))
    #print(data_raw['Mileage'].unique())
    #check for expensive cars that can be misinputs
    print(data_raw[data_raw['Price'] > 1000000].sort_values('Year')[['Brand','Model','Mileage','Equipment','Origin','First Owner','Year','Price']])
    #check again but for more specific inputs
    data_raw[(data_raw['Price'] > 1000000) & (data_raw['Year'].astype(int) >= 2010)].sort_values('Year')[['Brand','Model','Mileage','Equipment','Origin','First Owner','Year','Price']].to_csv('../data/raw/outliers_stage3.csv')
    #look through values to know that there for sure aren't any Nan's and misinputs
    print(data_raw['Condition'].unique())
    print(data_raw['First Owner'].unique())
    print(data_raw['Number of Doors'].unique())
    print(data_raw['Fuel'].unique())
    #look for misinputs in brand
    print(data_raw['Brand'].value_counts().tail(20))
    data_raw.to_csv('../data/processed/clean_data',index=False)

