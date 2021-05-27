from preprocess import myPreprocess
from dataloader import Dataloader
from util import *
from model import *

if __name__ == '__main__':

    ############                       数据预加载（处理过缺失值）                 ############
    N_rows = 15000
    parse_dates = [['Date', 'Time']]
    filename = "../dataset/example_air_passengers.csv"
    df = preprocess(N_rows, parse_dates, filename)  #这里预处理加载的数据只专门针对第一列是时间格式为"1949-01-01"的数据
    columns = df.columns

    ############                       模型选择                 ############
    print("模型名称如下：GM，Arima，LightGBM，Xgb")
    modelName = input("请输入模型名称：")
    funcName = "predict"

    ############                       特征选择                 ############
    print("数据特征如下：")
    map = dict()
    for i in range(0, len(columns)):
        map[str(i)] = columns[i]
        print(str(i) + "：" + columns[i])
    featureInput = input("请选择特征（例如1,2,3..)：")
    labelInput = input("请选择标签（只能选一个）：")

    featureCols = featureInput.split(",")
    labelCol = labelInput
    print("featureCols = {},labelCol = {}".format(featureCols, labelCol))

    # 将特征编号,标签编号转化成特征名，标签编号
    featureNames = []
    for i in featureCols:
        featureNames.append(map[i])
    labelName = map[str(labelCol)]
    # print(featureNames)

    #############                  数据预处理             #############
    df = myPreprocess.myPreprocess(df,modelName)

    #############                  划分训练集和测试集             #############
    if(modelName == "LightGBM"):
        X_train, X_test, y_train, y_test = Dataloader.DataLoader(df,featureNames,labelName)   #GM，Arima无需划分训练集和测试集


    #############                  模型的训练             #############
    module = __import__("model",globals(),locals(),[modelName])  #获取模块
    pyModel = getattr(module,modelName)  #获取模块中python文件
    modelCls = getattr(pyModel,modelName)  #获取模型
    model = modelCls()

    if(modelName == "GM"):
        '''GM模型无需训练模型，故直接输入整个一维时间序列即可'''
        # 模型参数设置
        X = df[df.columns[0]]
        print(X)
        predict = model.predict(X,20)
        print(predict)
        pass

    elif(modelName == "LightGBM"):
        # 模型参数设置
        model.fit(X_train,y_train,X_test,y_test)
        predict = model.predict(100)
        print(predict)
        pass

    elif(modelName == "Xgb"):
        # 模型参数设置
        predict = model.predict(df) #模型预测
        print(predict)
        pass

    elif(modelName == "Arima"):

        # 模型参数设置

        # 模型预测
        G_power = pd.to_numeric(df["Global_active_power"])
        # time series plot of one-minute sampling rate data
        timeseries_plot(G_power, 'g', 'Global_active_power')

        # we take a 30 minutes bucket average of our time series data to reduce noise.
        bucket_size = "30T"
        G_power_avg = bucket_avg(G_power, bucket_size)
        # plot of 30 minutes average.
        ts_label = 'G_power_avg'
        timeseries_plot(G_power_avg, 'g', ts_label)

        # "Grid search" of seasonal ARIMA model.
        # the seasonal periodicy 24 hours, i.e. S=24*60/30 = 48 samples
        arima_para = {}
        arima_para['p'] = range(2)
        arima_para['d'] = range(2)
        arima_para['q'] = range(2)
        # the seasonal periodicy is  24 hours
        seasonal_para = round(24 * 60 / (float(bucket_size[:-1])))
        arima = model
        arima.paramSetter(arima_para,seasonal_para)   #给模型添加参数

        arima.fit(G_power_avg)

        # Forecasts to unseen future data
        n_steps = 100  # next 100 * 30 min = 50 hours
        predict = arima.predict(n_steps)
        print(predict)
        pass

    #############                  模型结果的可视化            #############

