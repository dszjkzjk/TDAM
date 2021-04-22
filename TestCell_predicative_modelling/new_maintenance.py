#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:15:53 2019

@author: junkangzhang
"""

import matplotlib.pyplot as plt
import numpy as np
import mpld3
import pandas as pd
import matplotlib.dates as mdates

#unsche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/unscheduled_maintainence_22.csv')
#past_sche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/scheduled_maintainence_22.csv')
#future_sche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/future_maintainence_22.csv')
"""
unsche = unsche[['Notification date','Equipment','Completion by date']]
past_sche = past_sche[['Notification date','Equipment','Completion by date']]
future_sche = future_sche[['Planned Date', 'Equipment']]

unsche = unsche.fillna('Unknown')
past_sche = past_sche.fillna('Unknown')
future_sche = future_sche.fillna('Unknown')

unsche.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/tc22/unsche.csv',index=False)
past_sche.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/tc22/past_sche.csv',index=False)
future_sche.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 3/tc22/future_sche.csv',index=False)
"""

def col_rename(df):
    df.columns = df.columns.str.replace(' ','')

def year_month_day(df):
    #k=df.columns.get_loc("ROW_C_DATE")
    df['Date'] = [k[:10] for k in df['ROW_C_DATE']]
    #df['Date'] = df['ROW_C_DATE']

def maintenance(tc_num,equipment,parameter,past=1,future=1,unsche=1):
    df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/df.csv')
    col_rename(df)
    df1 = df[['ROW_C_DATE',parameter]]
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
    year_month_day(df1)
    df1 = df1.drop('ROW_C_DATE',axis=1)
    
    if '-' not in df['ROW_C_DATE'].loc[0]:
        df['ROW_C_DATE'] = pd.to_datetime(df['ROW_C_DATE'],format='%m/%d/%Y %H:%M').dt.strftime('%Y-%m-%d %H:%M')
    if past == 1:
        past_sche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/past_sche.csv')
        past_sche_1 = past_sche[past_sche.Equipment == equipment]
    else: 
        past_sche_1 == 'Past Maintenance Schedules are not Requested'
    if future == 1:
        future_sche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/future_sche.csv')
        future_sche_1 = future_sche[future_sche.Equipment == equipment]
    else: 
        future_sche_1 == 'Future Maintenance Schedules are not Requested'
    if unsche == 1:
        unsche = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/BUSA693/Capstone/Individual/Individual 4/tc'+str(tc_num)+'/unsche.csv')
        unsche_1 = unsche[unsche.Equipment == equipment]
    else: 
        unsche_1 == 'Unscheduled Maintenance Records are not Requested'
    
    if isinstance(past_sche_1, pd.DataFrame):
        past_sche_2 = past_sche_1[['Completion by date']]
        past_sche_2.columns = ['Date']
        past_sche_3 = pd.merge(past_sche_2,df1,on='Date')
        if len(past_sche_3) == 0:
            past_sche_3 = past_sche_2
            past_sche_3[parameter] = [1 for i in range(len(past_sche_3))]
    else:
        past_sche_3 = pd.DataFrame(data={'Date': [], parameter: []})
        
    if isinstance(future_sche_1, pd.DataFrame):
        future_sche_2 = future_sche_1[['Planned Date']]
        future_sche_2.columns = ['Date']
        future_sche_3 = pd.merge(future_sche_2,df1,on='Date')
        if len(future_sche_3) == 0:
            future_sche_3 = future_sche_2
            future_sche_3[parameter] = [1 for i in range(len(future_sche_3))]
    else:
        future_sche_3 = pd.DataFrame(data={'Date': [], parameter: []})
        
    if isinstance(unsche_1, pd.DataFrame):
        unsche_2 = unsche_1[['Completion by date']]
        unsche_2.columns = ['Date']
        unsche_2 = unsche_2.reset_index(drop=True)
        if '-' not in unsche_2['Date'].loc[0]:
            unsche_2.Date = pd.to_datetime(unsche_2.Date,format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
        unsche_3 = pd.merge(unsche_2,df1,on='Date')
        if len(unsche_3) == 0:
            unsche_3 = unsche_2
            unsche_3[parameter] = [1 for i in range(len(unsche_3))]
    else:
        unsche_3 = pd.DataFrame(data={'Date': [], parameter: []})
            
    #plot
    past_sche_3 = past_sche_3.groupby('Date')[parameter].max()
    future_sche_3 = future_sche_3.groupby('Date')[parameter].max()
    unsche_3 = unsche_3.groupby('Date')[parameter].max()
    df_plot1 = pd.concat([unsche_3,past_sche_3,future_sche_3],axis=1)
    df_plot1.index = pd.to_datetime(df_plot1.index,format='%Y-%m-%d')
    df_plot1.columns = ['unscheduled','past scheduled','future scheduled']
    df_plot1.plot(style='o',title="Test Cell 22 " + str(parameter) + ' maintenance schedules')
    
"""        
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
N = 100

scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s=1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)

ax.grid(color='white', linestyle='solid')
ax.set_title("Scatter Plot (with tooltips!)", size=20)

labels = ['point {0}'.format(i + 1) for i in range(N)]
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.show()
"""
maintenance(22,'M98122-18','T.FAN1')
