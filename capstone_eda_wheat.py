# -*- coding: utf-8 -*-
"""
The purpose of this script is to prepare the wheat data for modeling.

@author: Caleb Waack
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# The data was downloaded from the internet in csv format

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

df_wheat = data[data['Commodity']=='Wheat']
df_wheat = df_wheat[df_wheat['Region'] == 'United States']

df_wheat = df_wheat[df_wheat['Unit'] != 'Years']
df_wheat = df_wheat[df_wheat['Unit'] != '$/bu']
df_wheat = df_wheat[df_wheat['Unit'] != 'Percent']
df_wheat = df_wheat[df_wheat['Unit'] != 'Bushels']

df_wheat['Quantity'] = np.where(df_wheat['Unit']=='Million Metric Tons',df_wheat['Value'],
                            np.where(df_wheat['Unit']=='Million Bushels',df_wheat['Value']*36.74,
                            np.where(df_wheat['Unit']=='Million Acres',df_wheat['Value']*1.361,
                                     0)))

df_wheat['Unit'] = 'Million Metric Tons'

df_wheat.drop(columns='Value',inplace=True)

#Exploring the data with a few visuals to get an understanding of the trends

df_wheat['Quantity'].hist()

sns.scatterplot(x='ForecastMonth',
                y='Quantity',
                hue='ForecastYear',
                data=df_wheat)
plt.show()

#After looking at the data on the day level, it was clear that the range was too large
#therefore I aggregated it to the month level, and wanted to see the trend 

df = df_wheat[['ForecastMonth','ForecastYear','Quantity']]
dfagg = df.groupby(['ForecastYear','ForecastMonth']).mean()
dfagg.reset_index(inplace=True)

sns.lineplot(x='ForecastMonth',
             y='Quantity',
             hue='ForecastYear',
                data=dfagg)
plt.show()

df['ForecastMonth'] = df['ForecastMonth'].astype(str)
df['ForecastYear'] = df['ForecastYear'].astype(str)
df['Date'] = df['ForecastYear'] + '-' + df['ForecastMonth']+'-'
df['Date'] = df['Date'].apply(lambda x: x + str(1))

df['Date'] = pd.to_datetime(df['Date'])

#Saving the file to a csv so that it would be viewable by others, and it can
#be uploaded into my modeling script

df.to_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Data/wheat_data_eda.csv')
