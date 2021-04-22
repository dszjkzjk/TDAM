#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:44:26 2019

@author: junkangzhang
"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from torch.autograd import Variable
import random
sns.set()
style='seaborn-paper'
#import math

### 1
df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/MGSC695-076/A1/pm2.5/FiveCitiePMData/ShanghaiPM20100101_20151231.csv')

#desc = data.describe()

### 2

### 3
# Pick target variable
df = df.drop(['PM_US Post', 'PM_Xuhui'],axis=1)
df = df[df['year']==2014]
#df = df[df['hour'].isin([7,15,23])]
# Delete NAs of the target variable
df = df.dropna(subset=['PM_Jingan'])
df = df.reset_index(drop=True)

# Dummify cbwd, month
dummy_cbwd = pd.get_dummies(df['cbwd'])
df = pd.concat([df,dummy_cbwd],axis=1)
dummy_month = pd.get_dummies(df['month'])

dummy_month.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df = pd.concat([df,dummy_month],axis=1)

"""
# Group df by year_month_day and take average of other varibales
df1 = df.groupby(['year','month','day'],as_index=False).mean()
df2 = df.groupby(['year','month','day'],as_index=False)['Iprec'].max()
df = df1
df['Iprec'] = df2['Iprec']
df = df.reset_index(drop=True)
"""
df = df.drop(['No','hour','season'],axis=1)


# Whether Weekday 
df['ymd'] = [str(df['year'].loc[i])+'-' + str(df['month'].loc[i])+'-' + str(df['day'].loc[i]) for i in range(len(df))]
df['date'] = [dt.datetime.strptime(d,'%Y-%m-%d') for d in df['ymd']]

weekday_lst = []
for i in df['date']:
    weekno = i.weekday()
    if weekno==6:
        weekday_lst.append('7')
    elif weekno==5:
        weekday_lst.append('6')
    elif weekno==4:
        weekday_lst.append('5')
    elif weekno==3:
        weekday_lst.append('4')
    elif weekno==2:
        weekday_lst.append('3')
    elif weekno==1:
        weekday_lst.append('2')
    else:
        weekday_lst.append('1')
df['week'] = weekday_lst

dummy_weekday = pd.get_dummies(df['week'])
dummy_weekday.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df = pd.concat([df,dummy_weekday],axis=1)

# Final Dataset
df = df.drop(['year','month','day','ymd','date','week','cbwd'],axis=1)
df_name = df.columns.tolist()
df_name = ['PM' if x=='PM_Jingan' else x for x in df_name]
df.columns = df_name
df = df.dropna()
#df = df.sample(frac=0.1)

### 4
from sklearn.metrics import r2_score
 
def calculate_r2(x,y):
    X = Variable(torch.FloatTensor(x))  
    result = model(X) 
    result=result.data
    r2=r2_score(result, y)
    return r2



class Net2(torch.nn.Module):
    def __init__(self, input_size, output_size,hidden_layers=(128,128)):
        super(Net2, self).__init__()
        self.first_layer=torch.nn.Linear(input_size, hidden_layers[0])
        self.final_layer = torch.nn.Linear(hidden_layers[-1], output_size)   # output layer
        self.hidden_layers=hidden_layers
        self.net=nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            self.net.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
    def forward(self, x):
        x = F.relu(self.first_layer(x))      # activation function for hidden layer
        for i in range(len(self.hidden_layers)-1):
            x=F.relu(self.net[i](x)) 
        x = self.final_layer(x)             # linear output
        return x

### 5
X = df.iloc[:,1:]
y = df['PM']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=0)

X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

### 6
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(torch.from_numpy(X_train).clone(), torch.from_numpy(y_train).clone())
train_loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)
dataset1 = TensorDataset(torch.from_numpy(X_test).clone(), torch.from_numpy(y_test).clone())
test_loader = DataLoader(dataset=dataset1, batch_size=50, shuffle=True)
## training
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

r2_training=[]
r2_test=[]

training_loss=[]
testing_loss=[]

lr = 0.001
reg= 0.001
model = Net2(X_train.shape[1], 1,hidden_layers=(100,100,100))
# model = Net(X_train.shape[1],1, 1 )

## optimal params: lr 5e-3, lambda= 5e-3 adam
## ratio training testing set =0.4
## architecture (50,100,50)
## architecture (100,100,100) gives better results

optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=reg)  ## define optimizer
# optimizer =torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()  # define loss function


for epoch in range(1001):
    running_loss = []
    for x,target in train_loader:
        inputs = Variable(torch.FloatTensor(x.float()))
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, torch.unsqueeze(target.float(),dim=1))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item()/len(inputs))
    training_loss.append(np.sqrt(np.mean(running_loss)))
        
    ## print statistics every 100 epochs
    if epoch%100==0:
        result1=calculate_r2(X_train,y_train)
        
        result2=calculate_r2(X_test,y_test)
        
        r2_training.append(result1)
        r2_test.append(result2)

        print('r2 score on training set ' +str(round(result1,2)) +' **** test set: '+str(round(result2,2)) + ' epoch '+str(epoch) )
        print('loss  epoch '+str(epoch)+' :'+str(loss.data))


model1 = Net2(X_train.shape[1], 1,hidden_layers=(100,100,100))
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr,weight_decay=reg)  ## define optimizer
# optimizer =torch.optim.SGD(model.parameters(), lr=0.01)
criterion1 = torch.nn.MSELoss()  # define loss function
for epoch in range(1000):
    running_loss1 = []
    for x,target in test_loader:
        inputs1 = Variable(torch.FloatTensor(x.float()))
        outputs1 = model1(inputs1)
        optimizer1.zero_grad()
        loss1 = criterion1(outputs1, torch.unsqueeze(target.float(),dim=1))
        loss1.backward()
        optimizer1.step()        
        running_loss1.append(loss1.item()/len(inputs1))
    testing_loss.append(np.sqrt(np.mean(running_loss1)))


# R2 RMSE
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,4))  

axes[0].plot(range(len(r2_training)),r2_training,label="training R2")
axes[0].plot(range(len(r2_training)),r2_test,label="testing R2")
axes[0].legend(loc=4,frameon=True,fontsize = 'medium').get_frame().set_facecolor('white')
axes[0].set_xlabel('Epoch',fontsize = 'large')
axes[0].set_ylabel('R2',fontsize = 'large')
axes[0].set_ylim((0,1 )) 
axes[0].set_title("R2 score")

axes[1].plot(range(len(training_loss)),training_loss,label="training loss")
axes[1].plot(range(len(testing_loss)),testing_loss,label="testing loss")
axes[1].legend(loc=0,frameon=True,fontsize = 'medium').get_frame().set_facecolor('white')
axes[1].set_xlabel('Epoch',fontsize = 'large')
axes[1].set_ylim((0,4 ))
axes[1].set_title("RMSE")


# predicted v.s actual
X_t = Variable(torch.FloatTensor(X_test))  
test_result = model(X_t) 
test_result=test_result.data

x = Variable(torch.FloatTensor(X_train))  
train_result = model(x) 
train_result=train_result.data

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,4))  
axes[0].scatter(test_result, y_test)
axes[0].plot([test_result.min(), test_result.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
axes[0].set_xlabel('Target (Ground truth)')
axes[0].set_ylabel('Price Predicted')
axes[0].set_title("Test set")


axes[1].scatter(train_result, y_train)
axes[1].plot([train_result.min(), train_result.max()], [y_train.min(), y_train.max()], 'k--', lw=3)
axes[1].set_xlabel('Target (Ground truth)')
axes[1].set_ylabel('Price Predicted')
axes[1].set_title("Training set")

### Calculate NN MSE
from sklearn.metrics import mean_squared_error
mse_nn = mean_squared_error(y_test, test_result)
mse_nn
r2_nn = max(r2_test)

### Compare with other methods
######################### Linear Regression
# Run linear regression
from sklearn.model_selection  import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train1,y_train1)

# Using the model to predict the results based on the test dataset
y_test1_pred = lm.predict(X_test1)

r2_lr = r2_score(y_test1_pred,y_test1)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse_lr = mean_squared_error(y_test1, y_test1_pred)

######################### K-NN

from sklearn.neighbors import KNeighborsRegressor
# cross validation to find with value of k(from 1 to 10) that provides the best prediction
lst = []
r2_lst = []
for i in range(1,11):
    knn = KNeighborsRegressor(n_neighbors=i)
    model = knn.fit(X_train,y_train1)
    y_test1_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test1, y_test1_pred)
    lst.append(mse)
    r2_knn = r2_score(y_test1_pred,y_test1)
    r2_lst.append(r2_knn)
    
mse_knn = min(lst)
r2_knn = max(r2_lst)


########################## Decision Tree
from sklearn.tree import DecisionTreeRegressor
lst3 = []
r2_lst = []

for i in range(1,11):
    dtree = DecisionTreeRegressor(max_depth=i)
    model = dtree.fit(X_train1,y_train1)
    y_test1_pred = dtree.predict(X_test1)
    mse = mean_squared_error(y_test1, y_test1_pred)
    lst3.append(mse)
    r2_dt = r2_score(y_test1_pred,y_test1)  
    r2_lst.append(r2_dt)

mse_dt = min(lst3)
r2_dt = max(r2_lst)


########################## Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0, n_estimators=100)
model = rf.fit(X_train1,y_train1)

# Using the model to predict the results based on the test dataset
y_test1_pred = rf.predict(X_test1)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse_rf = mean_squared_error(y_test1, y_test1_pred)
r2_rf = r2_score(y_test1_pred,y_test1)

### 8
# 2 hidden layer
# 3 hidden layer
# 4 hidden layer
## training
hidden_layers_size = [(100,100),(100,100,100),(100,100,100,100)]
r2_test_list = []
for i in hidden_layers_size:
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=0)
    
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(torch.from_numpy(X_train).clone(), torch.from_numpy(y_train).clone())
    train_loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    r2_training=[]
    r2_test=[]
    
    training_loss=[]
    #testing_loss=[]
    
    lr = 0.001
    reg= 0.001
    model = Net2(X_train.shape[1], 1,hidden_layers=i)
    # model = Net(X_train.shape[1],1, 1 )
    
    ## optimal params: lr 5e-3, lambda= 5e-3 adam
    ## ratio training testing set =0.4
    ## architecture (50,100,50)
    ## architecture (100,100,100) gives better results
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=reg)  ## define optimizer
    # optimizer =torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # define loss function
    
    
    for epoch in range(1001):
        running_loss = []
        for x,target in train_loader:
            inputs = Variable(torch.FloatTensor(x.float()))
        
            outputs = model(inputs)
            
            optimizer.zero_grad()
    
            loss = criterion(outputs, torch.unsqueeze(target.float(),dim=1))
            loss.backward()
            optimizer.step()
    
            
            running_loss.append(loss.item()/len(inputs))
        
        training_loss.append(np.sqrt(np.mean(running_loss)))
        
        
        ## print statistics every 100 epochs
        if epoch%100==0:
            result1=calculate_r2(X_train,y_train)
            
            result2=calculate_r2(X_test,y_test)
            
            r2_training.append(result1)
            r2_test.append(result2)
    
            print('r2 score on training set ' +str(round(result1,2)) +' **** test set: '+str(round(result2,2)) + ' epoch '+str(epoch) )
            print('loss  epoch '+str(epoch)+' :'+str(loss.data))

    r2_test_list.append(r2_test)
    
fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8,4))  
axes.plot(range(len(r2_training)),r2_training,label="training R2")
axes.plot(range(len(r2_training)),r2_test_list[0],label="testing R2, 2-layer")
axes.plot(range(len(r2_training)),r2_test_list[1],label="testing R2, 3-layer")
axes.plot(range(len(r2_training)),r2_test_list[2],label="testing R2, 4-layer")
axes.legend(loc=1,frameon=True,fontsize = 'medium').get_frame().set_facecolor('white')
axes.set_xlabel('Epoch',fontsize = 'large')
axes.set_ylabel('R2',fontsize = 'large')
axes.set_ylim((0,1)) 
axes.set_title("R2 score")

lambda_list = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]
weights_list = []
for i in lambda_list:
    r2_training=[]
    r2_test=[]
    training_loss=[]
    testing_loss=[]
    lr = i
    reg= 0.001
    model = Net2(X_train.shape[1], 1,hidden_layers=(100,100,100,100,100))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=reg)  ## define optimizer
    # optimizer =torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # define loss function
    for epoch in range(101):
        running_loss = []
        for x,target in train_loader:
            inputs = Variable(torch.FloatTensor(x.float()))
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, torch.unsqueeze(target.float(),dim=1))
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()/len(inputs))
        training_loss.append(np.sqrt(np.mean(running_loss)))
    model_info = dict(model.state_dict())
    # final_layer's info
    final_layer_info = model_info['final_layer.weight']
    final_weight_array = final_layer_info.detach().numpy()
    final_weight = final_weight_array.tolist()
    weights_list.append(sum(final_weight[0])/len(final_weight[0]))
    print(i)
