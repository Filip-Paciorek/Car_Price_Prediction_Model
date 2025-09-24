import pandas as pd
data_raw = pd.read_csv('../data/raw/Moroccan Used Cars Dataset (MUCars - 2024)/cars_dataframe.csv')
if __name__ == "__main__":
    print(data_raw.head())
    for column in data_raw:
        #print(data_raw[column].unique())
        ...
    #print(data_raw.nunique())
    data_raw['Brand'] = data_raw['Brand'].str.lower()
    #print(data_raw.nunique())
    print(data_raw.isnull().sum())
    print(data_raw.shape)
    test = data_raw.dropna()
    print(test.shape)
    condition_check = test[['Condition','Price']]
    condition_check = condition_check.groupby('Condition')
    for i in condition_check:
        print(i)
    (test.loc[test['Year'].str.len() > 4,'Year']) = test.loc[test['Year'].str.len() > 4,'Year'].str.slice(0,4)     
    
    print(test[test['Price'] > 1000000].sort_values('Year')[['Brand','Model','Mileage','Equipment','Origin','First Owner','Year','Price']])
    test[(test['Price'] > 1000000) & (test['Year'].astype(int) >= 2010)].sort_values('Year')[['Brand','Model','Mileage','Equipment','Origin','First Owner','Year','Price']].to_csv('../data/raw/outliers_stage3.csv')
    print(test['Condition'].unique())
    print(test['First Owner'].unique())
    print(test['Number of Doors'].unique())
    print(test['Fuel'].unique())
    print(test['Brand'].value_counts().tail(20))
    #print(test['Model'].count())
     
