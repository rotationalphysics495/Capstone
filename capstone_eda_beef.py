# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:26:55 2022

@author: caleb
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

path = r"C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Raw Data"

file_list = os.listdir(path)
counter = 1

for ii in file_list:
    filepath = path +'/' + str(ii)
    
    df = pd.read_csv(filepath)
    
    if counter == 1:
        data = df[['Region','Commodity','Value','Unit','ReleaseDate',
                    'ForecastYear','ForecastMonth']]
        counter += 1
    else:   
        data = data.append(df[['Region','Commodity','Value','Unit','ReleaseDate',
                               'ForecastYear','ForecastMonth']])

df_beef = data[data['Commodity']=='Beef']
df_beef = df_beef[df_beef['Region'] == 'United States']

df_beef = df_beef[df_beef['Unit'] == 'Million Pounds']

df_beef['Value'].hist()

sns.scatterplot(x='ForecastMonth',y='Value',hue='ForecastYear',data=df_beef)
plt.show()

df = df_beef[['ForecastMonth','ForecastYear','Value']]
dfagg = df.groupby(['ForecastYear','ForecastMonth']).mean()
dfagg.reset_index(inplace=True)

sns.lineplot(x='ForecastMonth',
             y='Value',
             hue='ForecastYear',
                data=dfagg)
plt.show()

df['ForecastMonth'] = df['ForecastMonth'].astype(str)
df['ForecastYear'] = df['ForecastYear'].astype(str)
df['Date'] = df['ForecastYear'] + '-' + df['ForecastMonth']+'-'
df['Date'] = df['Date'].apply(lambda x: x + str(np.random.randint(1,28)))

df['Date'] = pd.to_datetime(df['Date'])

df.to_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Data/beef_data_eda.csv')

