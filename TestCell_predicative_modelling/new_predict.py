#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:27:15 2019

@author: junkangzhang
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
#from plotly.plotly import plot_mpl
#import plotly.plotly as ply
#import cufflinks as cf

df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 2/cleaned_22_data(1).csv')

def col_rename(df):
    df.columns = df.columns.str.replace(' ','')

def year_month_day(df):
    #k=df.columns.get_loc("ROW_C_DATE")
    df['Date'] = [k[:10] for k in df['ROW_C_DATE']]
    #df['Date'] = df['ROW_C_DATE']
    

"""
col_rename(df)
df1 = df[['ROW_C_DATE','T.FAN1']]
df1 = df1.dropna()
year_month_day(df1)
df1.Date = pd.to_datetime(df1.Date,format='%Y-%m-%d')
df1 = df1.groupby('Date')['T.FAN1'].max()
df1 = df1[df1 >=200]
df1  = df1.resample('14d').max().ffill()

# Run the AutoRegressive modelt
ts = df1[1:]
ar1 = AR(ts)
model1 = ar1.fit()

ts_pred_ar = model1.predict(start=len(ts),end=int(len(ts)*5/4),dynamic=False)

# Plot the graph comparing the real value and predicted value
ts_pred_ar[0] = ts[-1]
pyplot.xticks(rotation=30)
pyplot.plot(ts,color='blue')
pyplot.plot(ts_pred_ar,color='red')
pyplot.show()
"""
## AR


def tc_ar_predict(tc_num,parameter,interval,future_period):
    df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/tc'+str(tc_num)+'/df.csv')
    col_rename(df)
    if parameter not in df.columns.tolist():
        print(str(parameter) + " not in the dataframe")
    else:
        df1 = df[['ROW_C_DATE',parameter]]
        df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
        year_month_day(df1)
        df1.Date = pd.to_datetime(df1.Date,format='%Y-%m-%d')
        df1 = df1.groupby('Date')[parameter].max()
        df1 = df1[df1 < df1.quantile(.98)]
        df1 = df1[df1 > df1.quantile(.4)]
        
        """
        # drop max
        for i in range(int(len(df1)*0.02)):
            max_index = df1[df1 == max(df1)].index.tolist()[0]
            df1 =  df1.drop(max_index)
        # drop min
        for i in range(int(len(df1)*0.02)):
            max_index = df1[df1 == min(df1)].index.tolist()[0]
            df1 =  df1.drop(max_index)

        # filter out low values
        df1 = df1[df1 >=max(df1)/3]
        """
        df1  = df1.resample(str(interval)+'d').max().ffill()
        # Run the AutoRegressive model
        ts = df1
        ar1 = AR(ts)
        model1 = ar1.fit()
        coef = future_period/interval/len(df1)
        #
        future_forecast = model1.predict(start=len(ts)-1,end=int(len(ts)*(1+coef)),dynamic=False)
        # Plot the graph comparing the real value and predicted value
        future_forecast[0] = ts[-1]    
        future_forecast = pd.Series(future_forecast,index = future_forecast.index,name = 'Prediction')
        df_plot = pd.concat([df1,future_forecast],axis=1)
        df_plot.columns = ['Past Values','Prediction']
        df_plot.plot()
        """
        ts_pred_ar = model1.predict(start=len(ts)-1,end=int(len(ts)*(1+coef)),dynamic=False)
        # Plot the graph comparing the real value and predicted value
        ts_pred_ar[0] = ts[-1]
        plt.title(label='Time Series Forcasting of ' + str(parameter))
        plt.xticks(rotation=30)
        blue_patch = mpatches.Patch(color='blue', label='Past Values')
        red_patch = mpatches.Patch(color='red', label='Future Values')
        plt.legend(handles=[blue_patch,red_patch])
        plt.plot(ts,color='blue')
        plt.plot(ts_pred_ar,color='red')
        plt.show()
        """

## AR TRAIN TEST

def tc_ar_train_test(tc_num,parameter,interval):
    df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/tc'+str(tc_num)+'/df.csv')
    col_rename(df)
    if parameter not in df.columns.tolist():
        print(str(parameter) + " not in the dataframe")
    else:
        df1 = df[['ROW_C_DATE',parameter]]
        df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
        year_month_day(df1)
        df1.Date = pd.to_datetime(df1.Date,format='%Y-%m-%d')
        df1 = df1.groupby('Date')[parameter].max()
        df1 = df1[df1 < df1.quantile(.98)]
        df1 = df1[df1 > df1.quantile(.4)]

        df1  = df1.resample(str(interval)+'d').max().ffill()

        # Run the AutoRegressive model
        ts = df1
        split_size = round(len(ts)*0.1)
        train,test=ts[0:len(ts)-split_size],ts[len(ts)-split_size:]
        
        ar1 = AR(train)
        model1 = ar1.fit()
        future_forecast = model1.predict(start=len(train),end=len(train)+len(test)-1,dynamic=False)
        print(test)
        print(future_forecast)
        # Plot the graph comparing the real value and predicted value
        plt.title(label='Time Series Forcasting of ' + str(parameter))
        future_forecast = pd.Series(future_forecast,index = test.index,name = 'Prediction')
        df_plot = pd.concat([test,future_forecast],axis=1)
        df_plot.columns = ['Test','Prediction']
        df_plot.plot()


### ARIMA

def tc_arima_predict(tc_num,parameter,interval,future_period):
    df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/df.csv')
    if '-' not in df['ROW_C_DATE'].loc[0]:
        df['ROW_C_DATE'] = pd.to_datetime(df['ROW_C_DATE'],format='%m/%d/%Y %H:%M').dt.strftime('%Y-%m-%d %H:%M')
    col_rename(df)
    if parameter not in df.columns.tolist():
        print(str(parameter) + " not in the dataframe")
    else:
        df1 = df[['ROW_C_DATE',parameter]]
        df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
        year_month_day(df1)
        df1.Date = pd.to_datetime(df1.Date,format='%Y-%m-%d')
        df1 = df1.groupby('Date')[parameter].max()
        #print(df1)
        df1 = df1[df1 < df1.quantile(.98)]
        df1 = df1[df1 > df1.quantile(.2)]
        df1  = df1.resample(str(interval)+'d').max().ffill()
        # ARIMA
        stepwise_model = auto_arima(df1, start_p=2, start_q=2,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1,trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        ts = df1
        
        ## get future index
        ar1 = AR(ts)
        model = ar1.fit()
        coef1 = future_period/interval/len(df1)
        ts_pred_ar = model.predict(start=len(ts)-1,end=int(len(ts)*(1+coef1)),dynamic=False)
        future_index = ts_pred_ar.index
        ## end
        
        model1 = stepwise_model.fit(ts)
        coef = future_period/interval
        future_forecast = model1.predict(n_periods = len(ts_pred_ar))
        future_forecast[0] = ts[-1]
        # Plot the graph comparing the real value and predicted value
        future_forecast = pd.Series(future_forecast,index = future_index,name='Prediction')
        df_plot = pd.concat([df1,future_forecast],axis=1)
        df_plot.columns = ['Past Values','Prediction']
        df_plot.plot()

        result1 = seasonal_decompose(ts,freq=3, model='additive')
        result2 = seasonal_decompose(future_forecast,freq=3, model='additive') 
        
        trend1 = result1.trend
        trend2 = result2.trend
        trend1 = trend1.fillna(method='ffill')
        trend2 = trend2.fillna(method='bfill')
        trend2[0]=trend1[-1]
        df_plot1 = pd.concat([trend1,trend2],axis=1)
        df_plot1.columns = ['Past Trend','Predictive Trend']
        df_plot1.plot()
        ## decompose
        #fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,8))
        #result.trend.plot(ax=ax1)
        #result.resid.plot(ax=ax2)
        #result.seasonal.plot(ax=ax3)
        #result2.plot()

        
### ARIMA TRAIN TEST

def tc_arima_train_test(tc_num,parameter,interval):
    df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/tc'+str(tc_num)+'/df.csv')
    col_rename(df)
    if parameter not in df.columns.tolist():
        print(str(parameter) + " not in the dataframe")
    else:
        df1 = df[['ROW_C_DATE',parameter]]
        df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
        year_month_day(df1)
        df1.Date = pd.to_datetime(df1.Date,format='%Y-%m-%d')
        df1 = df1.groupby('Date')[parameter].max()
        df1 = df1[df1 < df1.quantile(.98)]
        df1 = df1[df1 > df1.quantile(.4)]

        df1  = df1.resample(str(interval)+'d').max().ffill()

        # ARIMA
        stepwise_model = auto_arima(df1, start_p=1, start_q=1,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        ts = df1
        split_size = round(len(ts)*0.2)
        train,test=ts[0:len(ts)-split_size],ts[len(ts)-split_size:]
        model1 = stepwise_model.fit(train)
        future_forecast = model1.predict(n_periods = len(test))
        # Plot the graph comparing the real value and predicted value
        future_forecast = pd.Series(future_forecast,index = test.index,name='Prediction')
        df_plot = pd.concat([test,future_forecast],axis=1)
        df_plot.columns = ['Test','Prediction']
        df_plot.plot()
        #pd.concat([df1,future_forecast],axis=1).plot()


