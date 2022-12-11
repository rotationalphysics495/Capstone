# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:59:26 2022

@author: caleb
"""

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

#Reading in the dataset
df = pd.read_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Data/wheat_data_eda.csv')

# Generating visuals that will be needed for the analysis
pd.set_option("display.max_columns", None)
sns.distplot(df['Quantity'],kde=False,bins=50).set(title='Histogram of Wheat Quantity per Day')
plt.savefig(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/wheat_hist_quantity_daily.pdf',
    format='pdf')
plt.show()

df.drop(columns=['Unnamed: 0','ForecastMonth','ForecastYear'],inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

#Prophet only takes columns with name y and ds
df.columns = ['y','ds']

#Setting up the hyperparameter grid for the grid search methodology
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }

all_params = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

#Loop for the performance metrics for the cross validation procedure 
counter = 0
for params in all_params:
    m = Prophet(**params).fit(df)
    df_cv = cross_validation(m,initial = '730 days',
                             period='182 days', horizon='365 days')
    if counter == 0:
        df_m = performance_metrics(df_cv,rolling_window=1)
        counter += 1
    else:
        df_p = performance_metrics(df_cv,rolling_window=1)
        df_metrics = pd.concat([df_p,df_m],ignore_index=True)
        df_m = df_metrics
        

#Daily predicitions model metrics
daily_pred_mm = df_m
daily_pred_mm['mape'] = 0
#Mape was set to 0 here because it was dropped for being too close to zero by the model

df = pd.read_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Data/wheat_data_eda.csv')

df.drop(columns=['Unnamed: 0','Date'], inplace = True)
df = df.groupby(['ForecastYear','ForecastMonth']).sum()
df.reset_index(inplace=True)

sns.distplot(df['Quantity'],
             kde=False,bins=50
             ).set(title='Histogram of Wheat Quantity per Month')
plt.savefig(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/wheat_hist_quantity_month.pdf',
    format='pdf')
plt.show()

#Aggregating the data for modeling on the month level

df['ForecastMonth'] = df['ForecastMonth'].astype(str)
df['ForecastYear'] = df['ForecastYear'].astype(str)
df['Date'] = df['ForecastYear'] + '-' + df['ForecastMonth']+'-'
df['Date'] = df['Date'].apply(lambda x: x + str(1))
                              
df['Date'] = pd.to_datetime(df['Date'])

df.drop(columns=['ForecastYear','ForecastMonth',],inplace = True)

df.columns = ['y','ds']

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }

all_params = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

counter = 0
for params in all_params:
    m = Prophet(**params).fit(df)
    df_cv = cross_validation(m,initial = '730 days',
                             period='182 days', horizon='365 days')
    if counter == 0:
        df_m = performance_metrics(df_cv,rolling_window=1)
        counter += 1
    else:
        df_p = performance_metrics(df_cv,rolling_window=1)
        df_metrics = pd.concat([df_p,df_m],ignore_index=True)
        df_m = df_metrics


# monthly predicitions model metrics
monthly_pred_mm = df_m

monthly_tuning_results = pd.DataFrame(all_params)
daily_tuning_results = pd.DataFrame(all_params)

#Creating a dataframe to compare the performance of the model on daily data
#with random noise induced, and monthly data

for ii in monthly_pred_mm.columns:
    monthly_tuning_results[ii] = monthly_pred_mm[ii]
    daily_tuning_results[ii] = daily_pred_mm[ii]
    
monthly_tuning_results.drop(
    columns=['horizon','mse','rmse','mae','coverage'],
    inplace=True
    )
daily_tuning_results.drop(
    columns=['horizon','mse','rmse','mae','coverage'],
    inplace=True
    )

#Loop to create the relevant model metrics vs hyperparameter values
for ii in monthly_tuning_results.columns:
    if ii == 'changepoint_prior_scale' or ii == 'seasonality_prior_scale':
        next
    else:
        sns.lineplot(x='changepoint_prior_scale',
                     y=ii,
                     data=monthly_tuning_results,
                     ci=False,
                     palette='coolwarm').set(
                         title=str(ii)+' vs Changepoint Prior Scale')
        filepath = (
            'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/' +
            str(ii)+'changepoint_lineplot.pdf'
            )
        plt.savefig(
            filepath,
            format='pdf')
        plt.show()
        sns.lineplot(x='seasonality_prior_scale',
                     y=ii,
                     data=monthly_tuning_results,
                     ci=False,
                     palette='coolwarm').set(
                         title=str(ii)+' vs Seasonality Prior Scale')
        filepath = (
            'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/' +
            str(ii)+'seasonality_lineplot.pdf'
            )
        plt.savefig(
            filepath,
            format='pdf')
        plt.show()

#Upon inspenction, values were selected for the cps and sps. If this model were
#to be deployed the values to the model should be assigned programatically 

prophet = Prophet(growth='linear',
                  seasonality_mode='additive',
                  changepoint_range=0.8,
                  changepoint_prior_scale=0.1,
                  seasonality_prior_scale=10,
                  n_changepoints=6)

prophet.fit(df)
df_cv = cross_validation(
    m,initial = '730 days',
    period='182 days',
    horizon='365 days')

df_p = performance_metrics(df_cv)

future = prophet.make_future_dataframe(periods=12,freq='M')
forecast = prophet.predict(future)
prophet.plot_components(forecast)
plt.show()

#Creating preds vs actuals graph
pred = pd.DataFrame(df_cv['yhat'])
pred['Data_Type'] = 'Prediction'
pred['Date'] = df_cv['ds']
pred.columns = ['Value','Data_Type','Date']
act = pd.DataFrame(df_cv['y'])
act['Data_Type'] = 'Actual'
act['Date'] = df_cv['ds']
act.columns = ['Value','Data_Type','Date']
df_viz = pd.concat([act,pred],ignore_index=True)
plt.figure(figsize=(15,8))
sns.lineplot(x='ds',
             y='y',
             data=df).set(title='Wheat Actuals Over Time')
plt.show()

plt.figure(figsize=(15,8))
fig = sns.lineplot(x='Date',
             y='Value',
             data=df_viz,
             hue='Data_Type',
             ci=False,
             palette='coolwarm').set(
                 title='Wheat Predictions vs Actuals')

plt.show()


df_p['Days'] = df_p['horizon'].dt.days
sns.lineplot(
    x='Days',
    y='mdape',
    data=df_p,
    palette='coolwarm').set(
        title='Mdape as Prediction Horizon Increases'
        )
plt.show()

df_wheat =df

df = pd.read_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Price Data/wheat_price_data.csv')

#Calculating price elasticity for periods
df['Year Number'] = df['Year Number'].astype(str)
df['Month Number'] = df['Month Number'].astype(str)
df['Date'] = df['Year Number'] + '-' + df['Month Number']+'-'
df['Date'] = df['Date'].apply(lambda x: x + str(1))
                              
df['Date'] = pd.to_datetime(df['Date'])
df.drop(columns=['Year Number','Month Number'],inplace=True)

df.columns = ['Price','Date']

df['Price_Change'] = df['Price'].diff(periods=1)
df['Price_Change'] = df['Price_Change'].abs()

sns.lineplot(x='Date',
              y='Price_Change',
              data=df,
              ci=False).set(title='Price Change of Wheat Over Time')
plt.show()

df['Percent_Price'] = df['Price_Change'] / df['Price']
df_wheat['Supply_Change'] = df_wheat['y'].diff(periods=1)
df['Percent_Change'] = df_wheat['Supply_Change'] / df_wheat['y']
df['Percent_Change'] = df['Percent_Change'].abs()

df['Price_Elasticity'] = df['Percent_Change'] / df['Percent_Price']

sns.lineplot(x='Date',
              y='Price_Elasticity',
              data=df).set(title='Price Elasticity of Wheat Over Time')
plt.show()

