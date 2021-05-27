#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import datetime
import matplotlib.pyplot as plt

# 模型特征
feature_list = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
# 类别型特征
category_feature_list = ['feature1', 'feature2']
# 模型参数
params = {
    'boosting_type': 'gbdt',
    # 'boosting': 'dart',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'learning_rate': 0.01,
    'num_leaves': 10,
    'max_depth': 5,
    'max_bin': 10,
    'min_data_in_leaf': 8,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 0,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'min_split_gain': 0,
    'boost_from_average': False,
    'is_unbalance': True,
    'num_trees': 1,
    'verbose': 0
}

if __name__ == '__main__':
    print("load data")
    data = pd.read_csv('data.tsv', sep='\t').round(decimals=4)
    X = data[feature_list].values
    y = data['label'].values
    # X = data.loc[:, feature_list]
    # y = data.loc[:, 'label']

    # 训练集测试集切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
    # 测试集验证集切分
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)
    print("数据转换")
    lgb_train = lgb.Dataset(data=X_train, label=y_train, categorical_feature=category_feature_list)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=category_feature_list)
    # lgb_train = lgb.Dataset(X_train, y_train)
    # lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    print("train starting")
    gbm = lgb.train(params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    # categorical_feature=category_feature_list,
                    num_boost_round=100,
                    early_stopping_rounds=50)
    # 最佳迭代次数
    print(gbm.best_iteration)
    # 保存模型
    gbm.save_model('../models/gbm.model')