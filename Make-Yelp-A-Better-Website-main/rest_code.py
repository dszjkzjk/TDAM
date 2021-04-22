#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:57:45 2018

@author: junkangzhang
"""
import pandas as pd
import numpy as np

################# from R
business_rest = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest.csv')
business_hotel = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_hotel.csv')
review = pd.read_csv('/Users/junkangzhang/Downloads/yelp_review.csv')

##################### Variation k-mean
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X1_std = scaler.fit_transform(business_rest[['stars','review_count']])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
model = kmeans.fit(X1_std)
labels = model.predict(X1_std)

# Plot cluster membership
from matplotlib import pyplot
pyplot.scatter(business_rest['stars'], business_rest['review_count'], c=labels, cmap='rainbow') 

cluster_list = labels.tolist()
cluster_df = pd.DataFrame({'cluster':cluster_list})
business_rest['cluster'] = cluster_df['cluster']

#### 540 canada high rating high review resturants
#business_rest_hrhs = business_rest[business_rest.is_open == 1][business_rest.is_ca == 1][business_rest.review_count >= 100][business_rest.stars >= 4]
#business_rest_hrhs_open_canada = business_rest_hrhs
#business_rest_hrhs_open_canada.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest_hrhs_open_canada.csv')

##### random select 1000 resturants with more than 100 review counts resturants
kkk = business_rest[business_rest.is_open == 1][business_rest.is_ca == 1]
business_rest_review100 = business_rest[business_rest.is_open == 1][business_rest.is_ca == 1][business_rest.review_count >= 100]
business_rest1000_review100 = business_rest_review100.sample(n=1000,random_state = 1)
business_rest1000_review100 = business_rest1000_review100.drop(['Unnamed: 0'],axis=1)
business_rest1000_review100_review = pd.merge(business_rest1000_review100,review,on='business_id')
business_rest1000_review100_review.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest1000_review100_review.csv')
business_rest1000_review100.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest1000_review100.csv')

#business_rest_hrhs1 = business_rest_hrhs_review.sample(n=500,random_state = 1)
#business_rest_hrhs.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest_hrhs.csv')
#business_rest_hrhs_review = pd.merge(business_rest_hrhs,review,on='business_id')

#business_rest_hrhs_review1 = business_rest_hrhs_review.sample(n=3000,random_state = 1)
#business_rest_hrhs_review.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest_hrhs_review.csv')

############################################### to R to pick key words and create categorical variables.

##############################################

################### to python to do data mining
########## import ca resturants with number of each key word
#business_review_cato_open_ca = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_review_cato_open_ca.csv')
########## import business_rest1000_review100_review_catgo from R
business_rest1000_review100_review_catgo = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest1000_review100_review_catgo.csv')

######### merge
business_rest1000_review100 = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest1000_review100.csv')
final_set1 = pd.merge(business_rest1000_review100[['business_id','stars']],business_rest1000_review100_review_catgo,on='business_id')
final_set1 = final_set1.drop('Unnamed: 0',axis=1)
#final_set1.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/final_set1.csv')
###### Feature Selection - random forest
X = final_set1.iloc[:,2:66]
y= final_set1['stars']
###### Random Forest
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0)

final_model1 = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(final_model1, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
features = pd.DataFrame(list(zip(X.columns,final_model1.feature_importances_)), columns = ['predictor','Gini coefficient'])
features_sort = features.sort_values(['Gini coefficient'],ascending=False)
print(features_sort.head(10))
print(features_sort.tail(10))
features_sort.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/features.csv')
############### mse
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 5)

from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state = 0)
model3 = randomforest.fit(X_train, y_train)
y_test_pred = final_model1.predict(X_test)

from sklearn.metrics import mean_squared_error
mse3 = mean_squared_error(y_test, y_test_pred)
print(mse3)


############### Accuracy_score
X = final_set1.iloc[:,2:66]
y2 = final_set1['stars'] * 2
y2 = y2.astype(int)

########################### Lasso
X = final_set1.iloc[:,2:66]
y= final_set1['stars']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 5)

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)

from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01, positive=True)

model = ls.fit(X_train_std,y_train)
pd.DataFrame(list(zip(X_train.columns,model.coef_)), columns = ['predictor','coefficient'])

"""
from sklearn.ensemble import RandomForestClassfier
randomforest = RandomForestClassfier(random_state=0)
final_model2 = randomforest.fit(X, y2)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(final_model2, threshold=0.05)
sfm.fit(X, y2)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
   
pd.DataFrame(list(zip(X.columns,final_model2.feature_importances_)), columns = ['predictor','Gini coefficient'])
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y2,test_size = 0.3, random_state = 5)

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state = 0)
final_model2 = randomforest.fit(X_train, y_train)
y_test_pred = final_model2.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_test_pred)
print(score)


###################
#df.append(df2)
#business_rest = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest.csv')
#business = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/yelp_business.csv')


############################ SVR
X = final_set1.iloc[:,2:66]
y= final_set1['stars']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 5)

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)


from sklearn.svm import SVR
svm = SVR(kernel='linear', epsilon=0.1)
model = svm.fit(X_train_std,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = svm.predict(X_test_std)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse5 = mean_squared_error(y_test, y_test_pred)
print(mse5)



######################### K-NN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# cross validation to find with value of k(from 1 to 10) that provides the best prediction
for i in range(1,11):
    lst = []
    knn = KNeighborsRegressor(n_neighbors=i)
    model = knn.fit(X_train_std,y_train)
    y_test_pred = knn.predict(X_test_std)
    mse = mean_squared_error(y_test, y_test_pred)
    lst.append(mse)
    print(mean_squared_error(y_test,y_test_pred))


mse2 = min(lst)
