 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:44:06 2019

@author: junkangzhang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Timestamp
import ggplot
import math
import matplotlib.dates as mdates
import datetime as dt


#df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 2/cleaned_22_data(1).csv')
#sche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 2/scheduled_maintainence_22.csv')
#unsche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 2/unscheduled_maintainence_22.csv')
#hi_hihi = pd.read_excel('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 2/sub/hi_hihi.xlsx')

#def nan_None(k):
    #k = k.where((pd.notnull(df)), None)
    
# Rename colnames
def col_rename(df):
    df.columns = df.columns.str.replace(' ','')
    
#### clean timestamp
## Name all date columns with ROW_C_DATE
def year_date(df):
    #k=df.columns.get_loc("ROW_C_DATE")
    df['year_date'] = [k[:4] for k in df['ROW_C_DATE']]
    
def year_month_day(df):
    #k=df.columns.get_loc("ROW_C_DATE")
    df['Date'] = [k[:10] for k in df['ROW_C_DATE']]
    #df['Date'] = df['ROW_C_DATE']
    
## Year_selecting

#def year_select(df,year):
    #df_new = df[df.year_date == str(year)]
    #return df_new

def date(time):
    return dt.datetime.strptime(time, '%Y-%m-%d').date()

def time_select(df,start_date,end_date):
    df['period'] = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in df['Date']]
    df['date'] = [dt.datetime.strptime(d,'%Y-%m-%d') for d in df['Date']]
    df_new = df[(date(start_date) <=df['period']) & (df['period'] <= date(end_date))]
    return df_new

## Find Hi
def find_hi(df,parameter):
    a = df[df.Name==parameter]
    if len(a) != 0:
        b=a.iloc[0,3]
        c=a.iloc[0,4]
        if str(b) != 'nan' and str(c) != 'nan':
            return [b,c]
        else:
            return 'Missing Hi or HIHI'
    else:
        return "Not in HI file"

"""        
## example
year_date(df)
year_month(df)
df2015 = year_select(df,2015)
"""

## Create status
def status(df,parameter,hi):
    lst = []
    for k in df[parameter]:
        if k>=hi[1]:
            lst.append('Shutdown')
        elif k>=hi[0]:
            lst.append('Alert')
        else:
            lst.append('Normal')
    return lst


def plot_insights(tc_num,start_date,end_date,parameter):
    df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/df.csv')
    if '-' not in df['ROW_C_DATE'].loc[0]:
        df['ROW_C_DATE'] = pd.to_datetime(df['ROW_C_DATE'],format='%m/%d/%Y %H:%M').dt.strftime('%Y-%m-%d %H:%M')
    if tc_num == 22:
        hihi = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/hihi.csv')
    col_rename(df)
    # mergeing parameters
    col_name = df.columns.tolist()
    matching = [s for s in col_name if parameter in s]
    matching = ['ROW_C_DATE'] + matching
    
    #if parameter not in df.columns.tolist():
    if len(matching) == 1:
        print(str(parameter) + " not in the dataframe")
    # Rename col
    elif len(matching) == 2:
        # Get hi
        if tc_num == 22:
            hi = find_hi(hihi,parameter)
        else:
            hi = 'Missing Hi or HIHI'
        # subset data by year
        year_month_day(df)
        df_sub = time_select(df,start_date,end_date)
        df_sub = df_sub[['GE_SN','GE_BUILD','CONDITION',parameter,'date']]
        # dropna
        df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna()
        #
        if hi != 'Missing Hi or HIHI' and hi != "Not in HI file":
            df_sub['status']=status(df_sub,parameter,hi)
            #df_sub['date'] = [dt.datetime.strptime(d,'%Y-%m-%d') for d in df_sub['Date']]
            #ggplot(aes(x='Date', y='T.FAN1', color='status'), data=df_tfan1) +geom_point(size=50) + scale_x_date()
            groups = df_sub.groupby('status')
            # Plot
            fig, ax = plt.subplots()
            ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
            for name, group in groups:
                ax.plot(group['date'], group[parameter], marker='o', linestyle='', ms=5, label=name)
            ax.legend()
            plt.title("Test Cell 22 " + str(parameter) + " from " + str(start_date) + " to " + str(end_date))
            plt.axhline(y=hi[0], color='r', linestyle='-')
            plt.axhline(y=hi[1], color='b', linestyle='-')
            plt.show()
        else:
            #df_sub['status']=status(df_sub,parameter,hi)
            #df_sub['date'] = [dt.datetime.strptime(d,'%Y-%m-%d') for d in df_sub['Date']]
            #ggplot(aes(x='Date', y='T.FAN1', color='status'), data=df_tfan1) +geom_point(size=50) + scale_x_date()
            #groups = df_sub.groupby('status')
            # Plot
            fig, ax = plt.subplots()
            ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
            #for name, group in groups:
                #ax.plot(group['date'], group[parameter], marker='o', linestyle='', ms=5, label=name)
            ax.plot(df_sub['date'],df_sub[parameter],marker='o', linestyle='', ms=5)
            ax.legend()
            plt.title("Test Cell "+ str(tc_num) + " " + str(parameter) + " from " + str(start_date) + " to " + str(end_date))
            plt.show() 
            print(hi)
    else:
        df1 = df[matching]
        df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
        year_month_day(df1)
        df1.Date = pd.to_datetime(df1.Date,format='%Y-%m-%d')
        df1 = df1.drop('ROW_C_DATE',axis=1)
        df1 = df1.groupby('Date').max()
        df1  = df1.resample('7d').max().ffill()
        df1.plot(title="Test Cell "+ str(tc_num) + " " + str(parameter) + " from " + str(start_date) + " to " + str(end_date)) 

        