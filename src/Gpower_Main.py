import pandas as pd
from util import timeseries_plot, bucket_avg, preprocess, config_plot
from model.myArima import *
from analysis import Arima_analysis,GM_analysis,XGboost_analysis,LightGBM_analysis

config_plot()

# we focus on the last 10 days data in Nov 2010
N_rows = 15000
parse_dates = [['Date', 'Time']]
filename = "../dataset/household_power_consumption.txt"
df = preprocess(N_rows, parse_dates, filename)

if __name__ == '__main__':
    mode = 2
    if(mode == 0):
        '''GM模型'''
        X_pred = GM_analysis.GM_analysis(df)
        print(X_pred)

    elif(mode == 1):
        '''Arima模型'''
        Arima_analysis.Arima_analysis(df)

    elif(mode == 2):
        '''XGboost模型'''
        XGboost_analysis.XGboost_analysis(df)

    elif(mode == 3):
        '''LightGBM模型'''
        LightGBM_analysis.LightGBM_analysis(df)
