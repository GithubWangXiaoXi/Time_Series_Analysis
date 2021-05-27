import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from util import timeseries_plot, bucket_avg, preprocess, config_plot

class Arima:

    def paramSetter(self,arima_para=None, seasonal_para=None):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                             for x in list(itertools.product(p, d, q))]

    def fit(self,ts):
        print("Arima fit")

        warnings.filterwarnings("ignore")
        results_list = []
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(ts,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()

                    print('ARIMA{}x{}seasonal - AIC:{}'.format(param,
                                                               param_seasonal, results.aic))
                    results_list.append([param, param_seasonal, results.aic])
                except:
                    continue
        results_list = np.array(results_list)
        lowest_AIC = np.argmin(results_list[:, 2])
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('ARIMA{}x{}seasonal with lowest_AIC:{}'.format(
            results_list[lowest_AIC, 0], results_list[lowest_AIC, 1], results_list[lowest_AIC, 2]))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        mod = sm.tsa.statespace.SARIMAX(ts,
                                        order=results_list[lowest_AIC, 0],
                                        seasonal_order=results_list[lowest_AIC, 1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.final_result = mod.fit()
        print('Final model summary:')
        print(self.final_result.summary().tables[1])
        print('Final model diagnostics:')
        self.final_result.plot_diagnostics(figsize=(15, 12))
        plt.tight_layout()
        # plt.savefig('model_diagnostics.png', dpi=300)
        plt.show()

    def predict(self,n_steps):
        print("Arima predict")

        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()

        print("pred_ci = ", pred_ci)


