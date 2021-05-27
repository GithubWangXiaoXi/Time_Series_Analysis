import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import  adfuller
from scipy import stats
from itertools import product
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error

#<https://blog.csdn.net/cxg1105553864/article/details/91693158>
# Load data
df = pd.read_csv('../../dataset/example_air_passengers.csv')
df.ds = pd.to_datetime(df.ds)
df.index = df.ds   #将时间列做为索引
df.drop(['ds'], axis=1, inplace=True)  #删除原有的时间列

# Resampling
df_month = df.resample('M').mean()
print("df_month.head() = ", df_month.head())

# 拆分预测集及验证集
df_month_test = df_month[-5:]
print(df_month_test.tail())
print('df_month_test', len(df_month_test))
df_month = df_month[:-5]
print('df_month', len(df_month))

# PLOTS
fig = plt.figure(figsize=[15, 7])
plt.suptitle('sales, mean', fontsize=22)

plt.plot(df_month.y, '-', label='By Months')
plt.legend()

# plt.tight_layout()
plt.show()

# 看趋势
plt.figure(figsize=[15, 7])
sm.tsa.seasonal_decompose(df_month.y).plot()
plt.show()
print("air_passengers test: p={}".format(adfuller(df_month.y)[1]))
# air_passengers test: p=0.996129346920727

# Box-Cox Transformations ts序列转换
df_month['y_box'], lmbda = stats.boxcox(df_month.y)
print("air_passengers test: p={}".format(adfuller(df_month.y_box)[1]))
# air_passengers test: p=0.7011194980409873

# Seasonal differentiation
# 季节性差分确定sax中m参数
df_month['y_box_diff'] = df_month['y_box'] - df_month['y_box'].shift(12)

# Seasonal differentiation
# 季节性差分确定sax中m参数
df_month['y_box_diff'] = df_month['y_box'] - df_month['y_box'].shift(12)


#下面进行模型选择及预测输出指标：
# SARIMAX参数说明
'''
   趋势参数：（与ARIMA模型相同）
   p：趋势自回归阶数。
   d：趋势差分阶数。
   q：趋势移动平均阶数。

   季节性参数：
   P：季节性自回归阶数。
   D：季节性差分阶数。
   Q：季节性移动平均阶数。
   m：单个季节期间的时间步数。
'''

# Initial approximation of parameters
Qs = range(0, 3)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D = 1
d = 1

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
# list参数列表
print('parameters_list:{}'.format(parameters_list))
print(len(parameters_list))

results = []
best_aic = float("inf")

for parameters in parameters_list:
    try:
        # SARIMAX 训练的时候用到转换之后的ts
        model = sm.tsa.statespace.SARIMAX(df_month.y_box, order=(parameters[0], d, parameters[1]),
                                          seasonal_order=(parameters[2], D, parameters[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', parameters)
        continue

    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = parameters
    results.append([parameters, model.aic])

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by='aic', ascending=True).head())
print(best_model.summary())
# Model:             SARIMAX(0, 1, 1)x(1, 1, 2, 12)

sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48)


# 下图是对残差进行的检验。可以确认服从正太分布，且不存在滞后效应。
best_model.plot_diagnostics(lags=30, figsize=(16, 12))

df_month2 = df_month_test[['y']]
# best_model.predict()  设定开始结束时间
# invboxcox函数用于还愿boxcox序列
df_month2['forecast'] = inv_boxcox(best_model.forecast(steps=5), lmbda)
plt.figure(figsize=(15, 7))
df_month2.y.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted Sales')
plt.show()

# 获取mse
print('mean_squared_error: {}'.format(mean_squared_error(df_month2.y, df_month2.forecast)))
