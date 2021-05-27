from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
from util import *
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor  # wrapper
import scipy.stats as st

class Xgb:

    def fit(self):
        print("Xgb fit")

    #写死的
    def predict(self,df):
        print("Xgb predict")

        ##############################################################################
        # we only focus on the last 18000 points for datetime information
        # Run xgboost on all features
        # get data

        # 放在预处理过程中
        encode_cols = ['Month', 'DayofWeek', 'Hour']

        df1 = df.copy(deep=True)  # 保留原来的df

        # keep all features
        df = date_transform(df, encode_cols)

        # base parameters
        xgb_params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',  # regression task
            'subsample': 0.80,  # 80% of data to grow trees and prevent overfitting
            'colsample_bytree': 0.85,  # 85% of features used
            'eta': 0.1,
            'max_depth': 10,
            'seed': 42}  # for reproducible results

        #odel—param
        val_ratio = 0.3
        ntree = 300
        early_stop = 50

        print('-----Xgboost Using All Numeric Features-----',
              '\n---inital model feature importance---')
        fig_allFeatures = xgb_importance(
            df, val_ratio, xgb_params, ntree, early_stop, 'All Features')
        plt.show()

        #############################################################################
        # xgboost using only datetime information
        bucket_size = "5T"
        df = df1  # 还原成原来的df
        G_power = df["Global_active_power"]

        df = pd.DataFrame(bucket_avg(G_power, bucket_size))
        df.dropna(inplace=True)
        df.iloc[-1, :].index  # last time step  #2010-11-26 21:00:00

        test_start_date = '2010-11-25 20:00:00'
        unseen_start_date = '2010-11-26 21:10:00'

        steps = 200 #model—param
        # get splited data
        df_unseen, df_test, df = xgb_data_split(
            df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols)
        print('\n-----Xgboost on only datetime information---------\n')

        dim = {'train and validation data ': df.shape,
               'TEST data ': df_test.shape,
               'forecasting data ': df_unseen.shape}
        print(pd.DataFrame(list(dim.items()), columns=['Data', 'dimension']))

        # train model
        Y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                          test_size=val_ratio,
                                                          random_state=42)

        X_test = xgb.DMatrix(df_test.iloc[:, 1:])
        Y_test = df_test.iloc[:, 0]
        X_unseen = xgb.DMatrix(df_unseen)

        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)
        watchlist = [(dtrain, 'train'), (dval, 'validate')]

        # Grid Search  model—param
        params_sk = {
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'seed': 42}

        skrg = XGBRegressor(**params_sk)

        skrg.fit(X_train, y_train)

        params_grid = {"n_estimators": st.randint(100, 500),
                       #                "colsample_bytree": st.beta(10, 1),
                       #                "subsample": st.beta(10, 1),
                       #                "gamma": st.uniform(0, 10),
                       #                'reg_alpha': st.expon(0, 50),
                       #                "min_child_weight": st.expon(0, 50),
                       #               "learning_rate": st.uniform(0.06, 0.12),
                       'max_depth': st.randint(6, 30)
                       }
        search_sk = RandomizedSearchCV(
            skrg, params_grid, cv=5, random_state=1, n_iter=20)  # 5 fold cross validation
        search_sk.fit(X, Y)

        # best parameters
        print("best parameters:", search_sk.best_params_);
        print(
            "best score:", search_sk.best_score_)

        # with new parameters
        params_new = {**params_sk, **search_sk.best_params_}

        model_final = xgb.train(params_new, dtrain, evals=watchlist,
                                early_stopping_rounds=early_stop, verbose_eval=True)

        print('-----Xgboost Using Datetime Features Only------',
              '\n---Grid Search model feature importance---')
        importance = model_final.get_fscore()
        importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
        fig1 = feature_importance_plot(importance_sorted, 'feature importance')
        plt.show()

        #############################################################################
        # Forcasting
        # prediction to testing data
        Y_hat = model_final.predict(X_test)
        Y_hat = pd.DataFrame(Y_hat, index=Y_test.index, columns=["predicted"])

        # predictions to unseen future data
        unseen_y = model_final.predict(X_unseen)
        forecasts = pd.DataFrame(
            unseen_y, index=df_unseen.index, columns=["forecasts"])

        # plot forcast results using grid search final model
        plot_start = '2010-11-24 00:00:00'
        print('-----Xgboost Using Datetime Features Only------',
              '\n---Forecasting from Grid Search---')
        forecasts_plot2 = xgb_forecasts_plot(
            plot_start, Y, Y_test, Y_hat, forecasts, 'Grid Search')

        # forcasts results using itinial model
        xgb_model = xgb.train(xgb_params, dtrain, ntree, evals=watchlist,
                              early_stopping_rounds=early_stop, verbose_eval=False)
        Y_hat = xgb_model.predict(X_test)
        Y_hat = pd.DataFrame(Y_hat, index=Y_test.index, columns=["test_predicted"])
        unseen_y = xgb_model.predict(X_unseen)
        forecasts = pd.DataFrame(
            unseen_y, index=df_unseen.index, columns=["forecasts"])
        plot_start = '2010-11-24 00:00:00'
        print('-----Xgboost Using Datetime Features Only------',
              '\n---Forecasting from initial---')
        forecasts_plot1 = xgb_forecasts_plot(
            plot_start, Y, Y_test, Y_hat, forecasts, 'Initial Model')

def xgb_data_split(df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols):
    # generate unseen data
    unseen = get_unseen_data(unseen_start_date, steps,
                             encode_cols, bucket_size)
    df = pd.concat([df, unseen], axis=0)
    df = date_transform(df, encode_cols)

    # data for forecast ,skip the connecting point
    df_unseen = df[unseen_start_date:].iloc[:, 1:]
    test_start = '2010-11-26 00:00:00'
    # skip the connecting point
    df_test = df[test_start_date: unseen_start_date].iloc[:-1, :]
    df_train = df[:test_start_date]
    return df_unseen, df_test, df_train

def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

def xgb_importance(df, test_ratio, xgb_params, ntree, early_stop, plot_title):
    df = pd.DataFrame(df)
    # split the data into train/TEST set
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'validate')]

    xgb_model = xgb.train(xgb_params, dtrain, ntree, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=True)

    importance = xgb_model.get_fscore()
    importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
    feature_importance_plot(importance_sorted, plot_title)


def xgb_forecasts_plot(plot_start, Y, Y_test, Y_hat, forecasts, title):
    Y = pd.concat([Y, Y_test])
    ax = Y[plot_start:].plot(label='observed', figsize=(15, 10))
    #Y_test.plot(label='test_observed', ax=ax)
    Y_hat.plot(label="predicted", ax=ax)
    forecasts.plot(label="forecast", ax=ax)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(Y_test.index[0]), Y_test.index[-1],
                     alpha=.1, zorder=-1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Global Active Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.show()
