# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
### import data
### validate data
### visulization
### feature selection
### machine learning
### cross-validation
### report

################################################################################################
########## Import Data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/individual/Dataset.xlsx')
df = df.iloc[:2000,:]

################################################################################################
##################################### Part 1 ###################################################
################### Summary Statistics
df_desc1 = df.describe()
df_desc2 = df.describe(include=['object'])

summary_stat = df_desc1[['usd_pledged','goal','blurb_len','name_len','launch_to_deadline_days']]
########## Each columns datatype
df.dtypes

################### Visualization

########## Variance-Covariance Matrix
numerics = ['int16', 'int32', 'int64', 'float16',    'float32', 'float64']
num_data = df.select_dtypes(include=numerics)
corr_matrix = num_data.corr()
sns.heatmap(corr_matrix, 
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns)

######### Distribution of states
states = df.groupby('state',as_index = False).count()
# Data to plot
labels = [states.state[i] for i in range(states.shape[0])]
sizes = [states.project_id[i] for i in range(states.shape[0])]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','lightgrey']
explode = (0.1,0, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

########## #Distribution of countries
countries = df.groupby('country',as_index = False).count()
import matplotlib.pyplot as plt
# Data to plot
labels = [countries.country[i] for i in range(countries.shape[0])]
sizes = [countries.project_id[i] for i in range(countries.shape[0])]

width = 1/1.5
plt.figure()
plt.bar(labels, sizes, width, color="blue",align='center')
plt.xlabel('Country')
plt.title('Number of Projects in Each Conuntry')
plt.show()

########## Goal in usd range and successful rate
goal_succ = df.assign(usd_goal = df.goal * df.static_usd_rate)
goal_succ = goal_succ[goal_succ.state == 'successful']
bins = [0,100,1000,10000,100000,1000000,10**8]
goal_succ = goal_succ.groupby(pd.cut(goal_succ['usd_goal'], bins=bins)).state.count()
goal_succ.plot(kind='bar')
plt.xlabel('USD Goal')
plt.ylabel('Number of Successes')
plt.title('USD Goal Range v.s Number of Successes')
#plt.xticks(goal_succ.index,rotation=90)
plt.show()

########## Launched year and number of successes
years= df[df.state == 'successful']
years = years.groupby('launched_at_yr',as_index = False)['state'].count()
years_total = df.groupby('launched_at_yr',as_index = False)['state'].count()
# Data to plot
labels = [years.launched_at_yr[i] for i in range(years.shape[0])]
sizes = [years.state[i]/years_total.state[i] for i in range(years.shape[0])]

width = 1/1.5
plt.figure()
plt.bar(labels, sizes, width, color="grey",align='center')
plt.xlabel('Launched Year')
plt.ylabel('Success Rate')
plt.title('Success Rate for Each Year')
plt.show()

########## name_len v.s. usd_pledged
names = df.groupby('name_len',as_index = False)['usd_pledged'].mean()
# Data to plot
labels = [names.name_len[i] for i in range(names.shape[0])]
sizes = [names.usd_pledged[i] for i in range(names.shape[0])]

width = 1/1.5
plt.figure()
plt.bar(labels, sizes, width, color="lightskyblue",align='center')
plt.xlabel('Length of Project Name')
plt.ylabel('Average of Pledge in USD')
plt.title('Name Length v.s. Average of Pledge in USD')
plt.show()


################################################################################################
##################################### Part 2 ###################################################

#################################### Develope a regression model
######################### Clean data
######## dummify
dummy_country = pd.get_dummies(df['country'])
dummy_category = pd.get_dummies(df['category'])
df_dummify = pd.concat([df,dummy_country],axis=1)
df_dummify = pd.concat([df_dummify,dummy_category],axis=1)
#dataset_dummies = dataset_dummies.drop(['category'], axis=1)

######## Create usd_goal
df_dummify = df_dummify.assign(usd_goal = df_dummify.goal * df_dummify.static_usd_rate)

######## Use usd_pledged as the dependent variable, discard some variables
X_y =df_dummify[['usd_pledged','usd_goal',
 'disable_communication',
 'staff_pick',
 'name_len',
 'blurb_len_clean',
 'deadline_month',
 'deadline_day',
 'deadline_hr',
 'created_at_month',
 'created_at_day',
 'created_at_hr',
 'launched_at_month',
 'launched_at_day',
 'launched_at_hr',
 'create_to_launch_days',
 'launch_to_state_change_days',
 'launch_to_deadline_days',
 'AT',
 'AU',
 'BE',
 'CA',
 'CH',
 'DE',
 'DK',
 'ES',
 'FR',
 'GB',
 'HK',
 'IE',
 'IT',
 'LU',
 'MX',
 'NL',
 'NO',
 'NZ',
 'SE',
 'SG',
 'US',
 'Academic',
 'Apps',
 'Blues',
 'Comedy',
 'Experimental',
 'Festivals',
 'Flight',
 'Gadgets',
 'Hardware',
 'Immersive',
 'Makerspaces',
 'Musical',
 'Places',
 'Plays',
 'Restaurants',
 'Robots',
 'Shorts',
 'Software',
 'Sound',
 'Spaces',
 'Thrillers',
 'Wearables',
 'Web',
 'Webseries']]

X_y = X_y.dropna() 

X_picked = X_y.iloc[:,1:]
y = X_y['usd_pledged']
######################### Feature Selection

########## Lasso
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_picked_std = scaler.fit_transform(X_picked)

from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01, positive=True)
model1 = ls.fit(X_picked_std,y)
features1 = pd.DataFrame(list(zip(X_picked.columns,model1.coef_)), columns = ['predictor','coefficient'])
features_sort1 = features1.sort_values(['coefficient'],ascending=False)
features_sort1

########## Random Forest
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0)

model2 = randomforest.fit(X_picked, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model2, threshold=0.05)
sfm.fit(X_picked, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X_picked.columns[feature_list_index])
    
features2 = pd.DataFrame(list(zip(X_picked.columns,model2.feature_importances_)), columns = ['predictor','Gini coefficient'])
features_sort2 = features2.sort_values(['Gini coefficient'],ascending=False)
features_sort2

###################################### Set the final model for part 2
X_y_1 =df_dummify[['usd_pledged','usd_goal',
 'staff_pick',
 'US',
 'Hardware',
 'Wearables',
 'created_at_day',
 'name_len',
 'launch_to_deadline_days',
 'create_to_launch_days'
 ]]

X_y_1 = X_y_1.dropna() 

X1 = X_y_1.iloc[:,1:]
y1 = X_y_1['usd_pledged']
###################################### MSE
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size = 0.25, random_state = 5)

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)

######################### Linear Regression
# Run linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = lm.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y_test, y_test_pred)
print(mse1)

######################### K-NN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# cross validation to find with value of k(from 1 to 10) that provides the best prediction
lst = []
for i in range(1,11):
    knn = KNeighborsRegressor(n_neighbors=i)
    model = knn.fit(X_train_std,y_train)
    y_test_pred = knn.predict(X_test_std)
    mse = mean_squared_error(y_test, y_test_pred)
    lst.append(mse)
    print(mean_squared_error(y_test,y_test_pred))


mse2 = min(lst)

########################## Decision Tree
from sklearn.tree import DecisionTreeRegressor
lst3 = []
for i in range(1,11):
    dtree = DecisionTreeRegressor(max_depth=i)
    model = dtree.fit(X_train,y_train)
    y_test_pred = dtree.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    lst3.append(mse)
    print(mse)
    
mse3 = min(lst3)

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




############################# Find the lowest MSE
mse_lst = [mse1,mse2,mse3,mse4,mse5]
min(mse_lst)
print('The lowest MSE is ' + str(min(mse_lst)) + ' from Linear Regression')
#### Linear - Easy to intepretate





################################################################################################
##################################### Part 3 ###################################################
### 
df_dummify3 = df_dummify[(df_dummify.state == 'successful') | (df_dummify.state == 'failed')]
df_dummify3['is_successful']=df['state'].apply(lambda x: 1 if x == 'successful' else 0)
#dummy_successful = pd.get_dummies(df['state'])
#df_dummify3 = pd.concat([df_dummify,dummy_successful],axis=1)

X_y4 =df_dummify3[['is_successful','usd_goal',
 'disable_communication',
 'staff_pick',
 'name_len',
 'blurb_len_clean',
 'deadline_month',
 'deadline_day',
 'deadline_hr',
 'created_at_month',
 'created_at_day',
 'created_at_hr',
 'launched_at_month',
 'launched_at_day',
 'launched_at_hr',
 'create_to_launch_days',
 'launch_to_deadline_days',
 'AT',
 'AU',
 'BE',
 'CA',
 'CH',
 'DE',
 'DK',
 'ES',
 'FR',
 'GB',
 'HK',
 'IE',
 'IT',
 'LU',
 'MX',
 'NL',
 'NO',
 'NZ',
 'SE',
 'SG',
 'US',
 'Academic',
 'Apps',
 'Blues',
 'Comedy',
 'Experimental',
 'Festivals',
 'Flight',
 'Gadgets',
 'Hardware',
 'Immersive',
 'Makerspaces',
 'Musical',
 'Places',
 'Plays',
 'Restaurants',
 'Robots',
 'Shorts',
 'Software',
 'Sound',
 'Spaces',
 'Thrillers',
 'Wearables',
 'Web',
 'Webseries']]

X_y4 = X_y4.dropna() 

X_picked = X_y4.iloc[:,1:]
y = X_y4['is_successful']
######################### Feature Selection

########## Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)

model2 = randomforest.fit(X_picked, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model2, threshold=0.05)
sfm.fit(X_picked, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X_picked.columns[feature_list_index])
    
features3 = pd.DataFrame(list(zip(X_picked.columns,model2.feature_importances_)), columns = ['predictor','Gini coefficient'])
features_sort3 = features3.sort_values(['Gini coefficient'],ascending=False)
features_sort3


###################################### Set the final model for part 3
X_y_3 =df_dummify3[['is_successful','usd_goal',
 'staff_pick','create_to_launch_days','launched_at_day',
 'name_len','Web','launched_at_month',
 'launch_to_deadline_days']]
X_y_3 = X_y_3.dropna() 

X3 = X_y_3.iloc[:,1:]
y3 = X_y_3['is_successful']

###################################### Accuracy Score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X3,y3,test_size = 0.3, random_state = 2)

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)

######################### Logistic Regression
# Run linear regression
from sklearn.linear_model import LogisticRegression
lr2 = LogisticRegression(random_state=0)
model2=lr2.fit(X_train_std,y_train)

# Perform cross validation
from sklearn import metrics
y_test_pred = lr2.predict(X_test_std)
a1 = metrics.accuracy_score(y_test, y_test_pred)
print(a1)

########################## Decision Tree
from sklearn.tree import DecisionTreeClassifier
lstd = []
for i in range(1,11):
    dtree = DecisionTreeClassifier(max_depth=i)
    model = dtree.fit(X_train,y_train)
    y_test_pred = dtree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_test_pred)
    lstd.append(score)
    print(score)

print('\n')
a2= max(lstd)
print(a2)


########################## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0, n_estimators=100)
model = rf.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = rf.predict(X_test)

# Calculate the mean squared error of the prediction
a3 = metrics.accuracy_score(y_test, y_test_pred)
print(a3)


############################# Find the highest accuracy score
score_lst = [a1,a2,a3]
min(score_lst)
print('The highest score is ' + str(min(score_lst)) + ' from Decision Tree at max_depth 3.')





################################################################################################
##################################### Part 4 ###################################################

####### cluster
from sklearn import datasets
import numpy


#X = df_dummify3[['usd_goal','US','launched_at_month']]
df_dummify4 = df_dummify.drop('launch_to_state_change_days',axis=1)
df_dummify4 = df_dummify4.dropna()
X = df_dummify4[['launch_to_deadline_days','create_to_launch_days']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X)
labels = model.labels_

from matplotlib import pyplot
pyplot.scatter(df_dummify4['launch_to_deadline_days'], df_dummify4['create_to_launch_days'], c=labels, cmap='rainbow') 


from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X,labels)

import pandas
df = pandas.DataFrame({'label':labels,'silhouette':silhouette})

print('Average Silhouette Score for Cluster 0: ',numpy.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',numpy.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',numpy.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',numpy.average(df[df['label'] == 3].silhouette))

from sklearn.metrics import silhouette_score
silhouette_score(X,labels)


df_dummify4['is_successful']=df_dummify4['state'].apply(lambda x: 1 if x == 'successful' else 0)

df_dummify4 = df_dummify4.assign(cluster = labels)
df_dummify4.groupby('cluster').count()

df_dummify4.groupby('cluster')['usd_goal'].mean()
df_dummify4.groupby('cluster')['usd_pledged'].mean()

a=df_dummify4[df_dummify4.state=='successful'].groupby('cluster')['is_successful'].count()
b=df_dummify4.groupby('cluster')['is_successful'].count()

a/b
