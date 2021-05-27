import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from matplotlib import pyplot as plt

class LightGBM:

    def __init__(self):
        # 将参数写成字典下形式
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'l2', 'auc'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        self.gbm = None
        self.num_boost_round = 20
        self.early_stopping_rounds = 5

    def paramSetter(self,**kwargs):
        if(kwargs['task'] != None): self.params['task'] = kwargs['task']
        if (kwargs['boosting_type'] != None): self.params['boosting_type'] = kwargs['boosting_type']
        if (kwargs['objective'] != None): self.params['objective'] = kwargs['objective']
        if (kwargs['metric'] != None): self.params['metric'] = kwargs['metric']
        if (kwargs['num_leaves'] != None): self.params['num_leaves'] = kwargs['num_leaves']
        if (kwargs['learning_rate'] != None): self.params['learning_rate'] = kwargs['learning_rate']
        if (kwargs['feature_fraction'] != None): self.params['feature_fraction'] = kwargs['feature_fraction']
        if (kwargs['bagging_fraction'] != None): self.params['bagging_fraction'] = kwargs['bagging_fraction']
        if (kwargs['bagging_freq'] != None): self.params['bagging_freq'] = kwargs['bagging_freq']
        if (kwargs['verbose'] != None): self.params['verbose'] = kwargs['verbose']
        if (kwargs['num_boost_round'] != None): self.num_boost_round = kwargs['num_boost_round']
        if (kwargs['early_stopping_rounds'] != None): self.early_stopping_rounds = kwargs['early_stopping_rounds']

    def fit(self,X_train,y_train,X_test,y_test):
        print("lightGBM fit")

        # 创建成lgb特征的数据集格式
        lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据
        self.valid_sets = lgb_eval

        print('Start training...')
        # 训练 cv and train
        gbm = lgb.train(self.params, lgb_train, self.num_boost_round, self.valid_sets,
                        self.early_stopping_rounds)  # 训练数据需要参数列表和数据集
        self.gbm = gbm

    def predict(self,K):
        '''
        预测长度为K的时间序列
        :return:
        '''
        print("lightGBM predict")

        print('Start predicting...')
        # 预测数据集
        y_pred = self.gbm.predict(X_test, num_iteration=self.gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
        # 评估模型
        print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差

        return y_pred