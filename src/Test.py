import pandas as pd
from model import Arima,GM,LightGBM,Xgb

df = pd.read_csv("../dataset/failureCount.csv")
columns = df.columns
print(columns)

ts = df["failureCount"].values
print(ts)
#Arima
model = Arima.Arima()
bucket_size = "30T"
arima_para = {}
arima_para['p'] = range(2)
arima_para['d'] = range(2)
arima_para['q'] = range(2)
# the seasonal periodicy is  24 hours
seasonal_para = round(24 * 60 / (float(bucket_size[:-1])))
model.paramSetter(arima_para=arima_para,seasonal_para=seasonal_para)
model.fit(ts)
model.predict(10)

#LightGBM
# model = LightGBM.LightGBM()
# model.fit()

#GM
model = GM.GM()
predict = model.predict(ts,100)
print(predict)

#XGboost



