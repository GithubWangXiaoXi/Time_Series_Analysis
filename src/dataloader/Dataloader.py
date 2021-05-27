from sklearn.model_selection import train_test_split
import pandas as pd
from util import timeseries_plot, bucket_avg, preprocess, config_plot

def DataLoader(df,featureNames,labelName,test_ratio = 0.3):

    df = pd.DataFrame(df)
    # split the data into train/TEST set
    X = df[featureNames]
    Y = df[[labelName]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    # 将df转化成ndarray并返回
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    return  X_train, X_test, y_train, y_test

if __name__ == '__main__':

    ############                       数据预处理                 ############
    N_rows = 15000
    parse_dates = [['Date', 'Time']]
    filename = "../../dataset/household_power_consumption.txt"
    df = preprocess(N_rows, parse_dates, filename)  #这里预处理加载的数据只专门针对household_power_consumption.txt(第一列数据是时间）
    columns = df.columns

    ############                       特征选择                 ############
    print("数据特征如下：")
    map = dict()
    for i in range(0,len(columns)):
        map[str(i)] = columns[i]
        print(str(i) + "：" + columns[i])
    featureInput = input("请选择特征（例如1,2,3..)：")
    labelInput = input("请选择标签（只能选一个）：")

    featureCols = featureInput.split(",")
    labelCol = labelInput
    print("featureCols = {},labelCol = {}".format(featureCols,labelCol))

    # 将特征编号,标签编号转化成特征名，标签编号
    featureNames = []
    for i in featureCols:
      featureNames.append(map[i])
    labelName = map[str(labelCol)]
    # print(featureNames)

    X_train, X_test, y_train, y_test = DataLoader(df,featureNames,labelName)
    print("X_train={}, X_test={}, y_train={}, y_test={}".format(X_train, X_test, y_train, y_test))