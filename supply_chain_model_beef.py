# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:43:31 2022

@author: caleb
"""

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Code/beef_data_eda.csv')

pd.set_option("display.max_columns", None)
sns.distplot(df['Value'],kde=False,bins=50).set(title='Histogram of Beef Quantity per Day')
plt.savefig(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/beef_hist_quantity_daily.pdf',
    format='pdf')
plt.show()

df.drop(columns=['Unnamed: 0','ForecastMonth','ForecastYear'],inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

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
    df_cvb = cross_validation(m,initial = '730 days',
                             period='182 days', horizon='365 days')
    if counter == 0:
        df_m = performance_metrics(df_cvb,rolling_window=1)
        counter += 1
    else:
        df_p = performance_metrics(df_cvb,rolling_window=1)
        df_metrics = pd.concat([df_p,df_m],ignore_index=True)
        df_m = df_metrics
        

# daily predicitions model metrics
daily_pred_mm = df_m

df = pd.read_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Code/beef_data_eda.csv')

df.drop(columns=['Unnamed: 0','Date'], inplace = True)
df = df.groupby(['ForecastYear','ForecastMonth']).sum()
df.reset_index(inplace=True)

sns.distplot(df['Value'],
             kde=False,bins=50
             ).set(title='Histogram of Beef Quantity per Month')
plt.savefig(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/beef_hist_quantity_month.pdf',
    format='pdf')
plt.show()

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
    df_cvb = cross_validation(m,initial = '730 days',
                             period='182 days', horizon='365 days')
    if counter == 0:
        df_m = performance_metrics(df_cvb,rolling_window=1)
        counter += 1
    else:
        df_p = performance_metrics(df_cvb,rolling_window=1)
        df_metrics = pd.concat([df_p,df_m],ignore_index=True)
        df_m = df_metrics


# monthly predicitions model metrics
monthly_pred_mm = df_m

monthly_tuning_results = pd.DataFrame(all_params)
daily_tuning_results = pd.DataFrame(all_params)

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

prophet = Prophet(growth='linear',
                  seasonality_mode='additive',
                  changepoint_range=0.8,
                  changepoint_prior_scale=0.1,
                  seasonality_prior_scale=10,
                  n_changepoints=6)

prophet.fit(df)
df_cvb = cross_validation(
    m,initial = '730 days',
    period='182 days',
    horizon='365 days')

df_pb = performance_metrics(df_cvb)

future = prophet.make_future_dataframe(periods=12,freq='M')
forecast = prophet.predict(future)
prophet.plot_components(forecast)
plt.show()

pred = pd.DataFrame(df_cvb['yhat'])
pred['Data_Type'] = 'Prediction'
pred['Date'] = df_cvb['ds']
pred.columns = ['Value','Data_Type','Date']
act = pd.DataFrame(df_cvb['y'])
act['Data_Type'] = 'Actual'
act['Date'] = df_cvb['ds']
act.columns = ['Value','Data_Type','Date']
df_viz_beef = pd.concat([act,pred],ignore_index=True)

plt.figure(figsize=(15,8))
fig = sns.lineplot(x='Date',
             y='Value',
             data=df_viz_beef,
             hue='Data_Type',
             ci=False,
             palette='coolwarm').set(
                 title='Beef Predictions vs Actuals')

plt.show()


df_pb['Days'] = df_pb['horizon'].dt.days
sns.lineplot(
    x='Days',
    y='mdape',
    data=df_pb,
    palette='coolwarm').set(
        title='Mdape as Prediction Horizon Increases'
        )
plt.show()

df_cvb.to_excel(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Price Data/beef_results.xlsx')

df_cvb['Supply_Change'] = df_cvb['y'].diff(periods=1)
df_cvb['Supply_Change'] = df_cvb['Supply_Change'].apply(abs)
df_cvb['Percent_Change'] = df_cvb['Supply_Change'] / df_cvb['y']

sns.lineplot(x='ds',
             y='Supply_Change',
             data=df_cvb,
             ci=False)
plt.show()

df_price = pd.read_csv(
    'C:/Users/caleb/OneDrive/Desktop/Personal/Data Science/Fall 2022/DS 785/Price Data/beef_price_data.csv')

df_price.columns = ['Date','Price']
df_price['Date'] = pd.to_datetime(df_price['Date'])
# converts price from cents to dollars per lb
df_price['Price'] = df_price['Price'] / 100
df_price['Price_Change'] = df_price['Price'].diff(periods=1)
df_price['Price_Change'] = df_price['Price_Change'].abs()

sns.lineplot(x='Date',
             y='Price_Change',
             data=df_price,
             ci=False).set(title='Price Change of Beef Over Time')
plt.ylim(0,.50)
plt.show()

df.columns = ['Value','Date']

df_price['Percent_Price'] = df_price['Price_Change'] / df_price['Price']
df['Supply_Change'] = df['Value'].diff(periods=1)
df['Supply_Change'] = df['Supply_Change'].abs()
df['Percent_Change'] = df['Supply_Change'] / df['Value']

df_price['Price_Elasticity'] = df['Percent_Change'] / df_price['Percent_Price']

sns.lineplot(x='Date',
             y='Price_Elasticity',
             data=df_price).set(title='Price Elasticity of Beef Over Time')
plt.ylim(0,1.2)
plt.show()