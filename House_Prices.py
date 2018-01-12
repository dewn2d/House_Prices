# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:12:33 2018

@author: Dwayne
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv("train.csv")
X_train = X_train.drop('SalePrice', axis = 1)
X_test  = pd.read_csv("test.csv")

y_train = pd.read_csv("train.csv")
y_train = y_train[['SalePrice']]

Id = pd.read_csv("test.csv")
Id = Id[['Id']]
I = Id.iloc[:,:].values

"""
with pd.option_context('display.max_rows', None, 'display.max_columns', 3): #print full list of columns         
    print(pd.isnull(X_train).sum() > 0) # print name of columns containing nan values
"""
   
drop_list = ["Id","Alley","PoolQC", "Fence", "MiscFeature"]
X_test = X_test.drop(drop_list, axis = 1)
X_train = X_train.drop(drop_list, axis = 1)


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with 0 value 
        in column.

        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else 0 for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
dfimputer =  DataFrameImputer() 
X_train = dfimputer.fit_transform( X_train )
X_test = dfimputer.transform( X_test )
         
with pd.option_context('display.max_rows', None, 'display.max_columns', 3): #print full list of columns         
    print(pd.isnull(X_train).sum() > 0) # print name of columns containing nan values
    print(X_test.dtypes)
    
# Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in X_train.columns.values: # Iterating over all the common columns in train and test
     if X_train[col].dtypes=='object':     # Encoding only categorical variables
         # Using whole data to form an exhaustive list of levels
         X_train[col]=le.fit_transform(X_train[col])
         X_test[col]=le.transform(X_test[col])
         
"""         
# Add deminsionality reduction

"""      


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
        
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 83 )
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Printing output file for kaggle
#compare current model with previous one

count = 0
prev_out = pd.read_csv('output.csv')
y_prev = prev_out.iloc[:, 1].values
for x in range(0,418):
    if y_prev[x] == y_pred[x]:
        count += 1
       
y_pred=np.matrix(y_pred)
y_pred = y_pred.T

output = np.concatenate((I, y_pred), axis = 1).astype(int)

df = pd.DataFrame(output,columns = ["Id","SalePrice"])
df.to_csv("output.csv", index = False )