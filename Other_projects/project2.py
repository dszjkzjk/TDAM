#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 01:12:22 2019

@author: junkangzhang
"""

def abs_mse(a,b):
    n = len(a)
    return sum(abs(a-b))/n

import pandas as pd
matlab = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/MRKT671/Project/results_fin_cute2.csv',header=None)
benz = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/MRKT671/Project/Mercedes Data.csv')
df = pd.concat([matlab,benz],axis=1)

######## dummify
dummy_model = pd.get_dummies(df['model'])
dummy_bg = pd.get_dummies(df['background_object'])
dummy_side = pd.get_dummies(df['is_side'])
dummy_head = pd.get_dummies(df['is_head'])
dummy_ratio = pd.get_dummies(df['ratio'])
dummy_color = pd.get_dummies(df['colour'])

df = pd.concat([df,dummy_model],axis=1)
df = pd.concat([df,dummy_bg],axis=1)
df = pd.concat([df,dummy_side],axis=1)
df = pd.concat([df,dummy_head],axis=1)
df = pd.concat([df,dummy_ratio],axis=1)
df = pd.concat([df,dummy_color],axis=1)

df = df.drop([0,'model','background_object','is_side','is_head','ratio','colour'],axis=1)

X = df.drop(['comments','likes'],axis=1)
y = df['likes']
#
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 5)

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)

########################## Feature Selection
########## Lasso
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.1, positive=True)
model1 = ls.fit(X_std,y)
features1 = pd.DataFrame(list(zip(X.columns,model1.coef_)), columns = ['predictor','coefficient'])
features_sort1 = features1.sort_values(['coefficient'],ascending=False)
features_sort1

########## Random Forest
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0)

model2 = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model2, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
features2 = pd.DataFrame(list(zip(X.columns,model2.feature_importances_)), columns = ['predictor','Gini coefficient'])
features_sort2 = features2.sort_values(['Gini coefficient'],ascending=False)
features_sort2


######################### Run linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = lm.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y_test, y_test_pred)
print(mse1)
absmse1 = abs_mse(y_test,y_test_pred)
######################### K-NN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# cross validation to find with value of k(from 1 to 10) that provides the best prediction
lst = []
d={}
for i in range(1,11):
    knn = KNeighborsRegressor(n_neighbors=i)
    model = knn.fit(X_train_std,y_train)
    y_test_pred = knn.predict(X_test_std)
    mse = mean_squared_error(y_test, y_test_pred)
    lst.append(mse)
    print(mean_squared_error(y_test,y_test_pred))
    d["absmse2_{0}".format(i)] = abs_mse(y_test,y_test_pred)

absmse2 = min(d.values())
print('\n')
mse2 = min(lst)
print(mse2)
k=lst.index(min(lst)) + 1
knn = KNeighborsRegressor(n_neighbors=k)
model = knn.fit(X_train_std,y_train)
y_test_pred = knn.predict(X_test_std)

########################## Decision Tree
from sklearn.tree import DecisionTreeRegressor
lst3 = []
dd = {}
for i in range(1,11):
    dtree = DecisionTreeRegressor(max_depth=i)
    model = dtree.fit(X_train,y_train)
    y_test_pred = dtree.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    lst3.append(mse)
    dd["absmse3_{0}".format(i)] = abs_mse(y_test,y_test_pred)
    print(mse)
    
absmse3 = min(dd.values())    
print('\n')
mse3 = min(lst3)
print(mse3)
k=lst3.index(min(lst3)) + 1
dtree = DecisionTreeRegressor(max_depth=k)
model = dtree.fit(X_train,y_train)
y_test_pred = dtree.predict(X_test)


########################## Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0, n_estimators=100)
model = rf.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = rf.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse4 = mean_squared_error(y_test, y_test_pred)
print(mse4)
absmse4 = abs_mse(y_test,y_test_pred)
############################ SVR
from sklearn.svm import SVR
svm = SVR(kernel='sigmoid', epsilon=0.1)
model = svm.fit(X_train_std,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = svm.predict(X_test_std)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse5 = mean_squared_error(y_test, y_test_pred)
print(mse5)
absmse5 = abs_mse(y_test,y_test_pred)
##### Min Mse
mse_lst = [mse1,mse2,mse3,mse4,mse5]
min(mse_lst)
k=mse_lst.index(min(mse_lst))
print(k)

minabs_mse = min(absmse1,absmse2,absmse3,absmse4,absmse5)
#### Define abs_mse

# KNN  


###############################
#features_sort1.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/MRKT671/Project/feature_sort1.csv')  

###############################
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
mean(benz['likes'])

amg = benz.groupby('is_AMG')['likes'].mean()
model = benz.groupby('model')['likes'].mean()
bg = benz.groupby('background_object')['likes'].mean()
exte = benz.groupby('exterior')['likes'].mean()
inte = benz.groupby('interior')['likes'].mean()




"""
###########################################
top30_feature = features_sort1.predictor[:30].tolist()
X = X[top30_feature]
y = df['likes']

######################### Run linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = lm.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y_test, y_test_pred)
print(mse1)
absmse1 = abs_mse(y_test,y_test_pred)
######################### K-NN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# cross validation to find with value of k(from 1 to 10) that provides the best prediction
lst = []
d={}
for i in range(1,11):
    knn = KNeighborsRegressor(n_neighbors=i)
    model = knn.fit(X_train_std,y_train)
    y_test_pred = knn.predict(X_test_std)
    mse = mean_squared_error(y_test, y_test_pred)
    lst.append(mse)
    print(mean_squared_error(y_test,y_test_pred))
    d["absmse2_{0}".format(i)] = abs_mse(y_test,y_test_pred)

absmse2 = min(d.values())
print('\n')
mse2 = min(lst)
print(mse2)
k=lst.index(min(lst)) + 1
knn = KNeighborsRegressor(n_neighbors=k)
model = knn.fit(X_train_std,y_train)
y_test_pred = knn.predict(X_test_std)

########################## Decision Tree
from sklearn.tree import DecisionTreeRegressor
lst3 = []
dd = {}
for i in range(1,11):
    dtree = DecisionTreeRegressor(max_depth=i)
    model = dtree.fit(X_train,y_train)
    y_test_pred = dtree.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    lst3.append(mse)
    dd["absmse3_{0}".format(i)] = abs_mse(y_test,y_test_pred)
    print(mse)
    
absmse3 = min(dd.values())    
print('\n')
mse3 = min(lst3)
print(mse3)
k=lst3.index(min(lst3)) + 1
dtree = DecisionTreeRegressor(max_depth=k)
model = dtree.fit(X_train,y_train)
y_test_pred = dtree.predict(X_test)


########################## Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0, n_estimators=100)
model = rf.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = rf.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse4 = mean_squared_error(y_test, y_test_pred)
print(mse4)
absmse4 = abs_mse(y_test,y_test_pred)
############################ SVR
from sklearn.svm import SVR
svm = SVR(kernel='sigmoid', epsilon=0.1)
model = svm.fit(X_train_std,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = svm.predict(X_test_std)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse5 = mean_squared_error(y_test, y_test_pred)
print(mse5)
absmse5 = abs_mse(y_test,y_test_pred)
##### Min Mse
mse_lst = [mse1,mse2,mse3,mse4,mse5]
min(mse_lst)
k=mse_lst.index(min(mse_lst))
print(k)
"""