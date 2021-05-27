import json
import lightgbm as lgbm
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from matplotlib import pyplot as plt
import numpy as np

class LightGBM:

    def __init__(self):
        # 将参数写成字典下形式
        self.params = {
            'max_depth' : 5,  # 最大深度
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'min_child_samples' : 80,
            'subsample' : 0.8,
            'colsample_bytree' : 1,
            'reg_alpha' : 0,
            'reg_lambda' : 0,
            'n_estimators' : 1000,
            'num_leaves': 25,  # 叶子节点数
            'learning_rate': 0.007,  # 学习速率
        }
        self.gbm = None
        self.eval_metric = 'l2',
        self.early_stopping_rounds = 200,
        self.verbose = False

    def paramSetter(self,**kwargs):
        if(kwargs['max_depth'] != None): self.params['max_depth'] = kwargs['max_depth']
        if (kwargs['boosting_type'] != None): self.params['boosting_type'] = kwargs['boosting_type']
        if (kwargs['objective'] != None): self.params['objective'] = kwargs['objective']
        if (kwargs['min_child_samples'] != None): self.params['min_child_samples'] = kwargs['min_child_samples']
        if (kwargs['subsample'] != None): self.params['subsample'] = kwargs['subsample']
        if (kwargs['colsample_bytree'] != None): self.params['colsample_bytree'] = kwargs['colsample_bytree']
        if (kwargs['reg_alpha'] != None): self.params['reg_alpha'] = kwargs['reg_alpha']
        if (kwargs['reg_lambda'] != None): self.params['reg_lambda'] = kwargs['reg_lambda']
        if (kwargs['n_estimators'] != None): self.params['n_estimators'] = kwargs['n_estimators']
        if (kwargs['learning_rate'] != None): self.params['learning_rate'] = kwargs['learning_rate']
        if (kwargs['num_leaves'] != None): self.params['num_leaves'] = kwargs['num_leaves']

        if(kwargs['eval_metric'] != None): self.eval_metric = kwargs['eval_metric']
        if (kwargs['early_stopping_rounds'] != None): self.early_stopping_rounds = kwargs['early_stopping_rounds']
        if (kwargs['verbose'] != None): self.verbose = kwargs['verbose']

    def fit(self,X_train,y_train,X_test,y_test):

        print("lightGBM fit")
        model = lgbm.LGBMRegressor(self.params["objective"], self.params["max_depth"], self.params["num_leaves"], self.params["learning_rate"],
                                   self.params["n_estimators"],self.params["min_child_samples"], self.params["subsample"], self.params["colsample_bytree"], self.params["reg_alpha"], self.params["reg_lambda"],
                                   random_state=np.random.randint(10e6))

        print("X_train = {}, y_train = {}".format(X_train.shape,y_train.shape))
        print(model)

        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_names=('fit', 'val'),
                  eval_metric = self.eval_metric, early_stopping_rounds = self.early_stopping_rounds, verbose = self.verbose)

        self.model = model
        self.predictX_start = X_train.shape[0] + X_test.shape[0]

    def predict(self,K):
        '''
        预测长度为K的时间序列
        :return:
        '''
        print("lightGBM predict")

        predictX = []
        for i in range(self.predictX_start,K,1):
            predictX.append(i)

        forecasttestY0 = self.model.predict(predictX)

        Hangnum = len(forecasttestY0)

        forecasttestY0 = np.reshape(forecasttestY0, (Hangnum, 1))
        print(forecasttestY0)


