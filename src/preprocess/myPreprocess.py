import pandas as pd
from util import timeseries_plot, bucket_avg, preprocess, config_plot

def myPreprocess(df,modelName):
    '''
    :param df: raw数据
    :param modelName: 模型名称
    :return:
    '''
    moduleName = "preprocess"
    funcStr = "preprocess_" + modelName

    #反射得到对应模型的预处理方法
    _temp = __import__(moduleName,globals(),locals(),["myPreprocess"])  #得到模块
    myPreprocess = _temp.myPreprocess  #得到模块中的python文件
    print("myPreprocess = ",myPreprocess)

    func = getattr(myPreprocess, funcStr)
    df = func(df)
    return df

def preprocess_GM(df):
    print("preprocess_GM")

    G_power = pd.to_numeric(df["Global_active_power"])
    # time series plot of one-minute sampling rate data
    timeseries_plot(G_power, 'g', 'Global_active_power')

    # we take a 30 minutes bucket average of our time series data to reduce noise.
    bucket_size = "30T"
    G_power_avg = bucket_avg(G_power, bucket_size)

    df = pd.DataFrame()
    df["G_power_avg"]  = G_power_avg
    return df

def preprocess_LightGBM(df):
    print("preprocess_LightGBM")
    return df

def preprocess_Xgb(df):
    print("preprocess_XGBoost")
    return df

def preprocess_Arima(df):
    print("preprocess_Arima")
    return df

if __name__ == '__main__':
    N_rows = 15000
    parse_dates = [['Date', 'Time']]
    filename = "../../dataset/household_power_consumption.txt"
    df = preprocess(N_rows, parse_dates, filename)
    df = myPreprocess(df,"Arima")
    print(df)