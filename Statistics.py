import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss as sm_kpss
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import pearsonr, spearmanr


class Statistics:
    """
    Performs statistics for Energy Forecaster
    """
    def __init__(self, ef):
        self._EF = ef
        self.datasets = None

    def mean(self, data):
        """
        Returns the mean of data without including nan values
        :param data: (numpy.ndarray) Data for calculations
        :return: (float) Mean of the data
        """
        return np.nanmean(data)

    def std(self, data, dof=1):
        """
        Returns the standard deviation of the data
        :param data: (numpy.ndarray) Data for calculations
        :param dof: (int) Degrees of freedom to use for the calculations
        :return: (float) Standard deviation of the data
        """
        return np.std(data, ddof=dof)

    def pearson_correlation(self, data_a, data_b):
        """
        Returns Pearson correlation for the two input datas
        :param data_a: (numpy.ndarray) First data
        :param data_b: (numpy.ndarray) Second data
        :return: (float) Pearson correlation for the data of the two input datas
        """
        return pearsonr(data_a, data_b)

    def spearman_correlation(self, data_a, data_b):
        """
        Returns Spearman correlation for the two input datas
        :param data_a: (numpy.ndarray) First data
        :param data_b: (numpy.ndarray) Second data
        :return: (float) Spearman correlation for the data of the two input datas
        """
        return spearmanr(data_a, data_b)

    def acf(self, data, nlags=None, qstat=False, alpha=None, missing='none'):
        """
        Calculates autocorrelation function
        :param data: (numpy.ndarray) Data to calculate ACF
        :param nlags: (int) Number of lags
        :param qstat: (bool) If True returns the Ljung-Box q statistic for each autocorrelation coefficient
        :param alpha: (float) For calculation of confident intervals
        :param missing: (str) Specifying how the nans are treated ('none', 'raise', 'conservative', 'drop')
        :return: (list) Results from statsmodels module acf
        """
        return sm_acf(data, nlags=nlags, qstat=qstat, alpha=alpha, missing=missing)

    def pacf(self, data, nlags=None, method='yw', alpha=None):
        """
        Estimates partial autocorrelation
        :param data: (numpy.ndarray) Data to estimate PACF
        :param nlags: (int) Number of lags
        :param method: Method for calculations ('yw', 'ywm', 'ols-inefficient', 'ols-adjusted', 'ld', 'ldb', 'burg')
        :param alpha: (float) For calculation of confident intervals
        :return: (list) Results from statsmodels module pacf
        """
        return sm_pacf(data, nlags=nlags, method=method, alpha=alpha)


class StatsData(Statistics):
    def kpss(self, data, nlags='auto', regression='c'):
        """
        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) unit root test from statsmodels module
        :param data: (numpy.ndarray) Data to apply KPSS unit root test
        :param nlags: (int) Number of lags to be used
        :param regression: (str) The null hypothesis for the KPSS test ('c', 'ct')
        :return: (list) KPSS test results from statsmodels module
        """
        return sm_kpss(x=data, regression=regression, nlags=nlags)

    def adf(self, data, maxlag=None, regression='c', autolag='AIC'):
        """
        Augmented Dickey-Fuller unit root test from statsmodels module
        :param data: (numpy.ndarray) Data to apply ADF unit root test
        :param maxlag: (int) Maximum lag which is included in test
        :param regression: (str) Constant and trend order to include in regression ('c', 'ct', 'ctt', 'n')
        :param autolag: (str) Method to use when automatically determining the lag length ('AIC', 'BIC', 't-stat', None)
        :return: (list) ADF test results from statsmodels module
        """
        return adfuller(x=data,
                        regression=regression,
                        maxlag=maxlag,
                        autolag=autolag)

    def zscore(self, data):
        """
        Calculates Z-score for a dataset
        :param data: (numpy.ndarray) Data to calculate Z-score
        :return: (numpy.ndarray) Z-score for the input data
        """
        mn = np.nanmean(data)
        std = np.nanstd(data)
        return (data - mn) / std

    def complexity_estimate(self, data):
        """
        Calculates Complexity Estimation
        :param data: (numpy.ndarray) Data to calculate Complexity Estimation
        :return: (float) Complexity Estimation
        """
        return np.sqrt(np.nansum(np.square(data[:-1] - data[1:])))

    def mean_absolute_change(self, data):
        """
        Calculates Mean Absolute Change
        :param data: (numpy.ndarray) Data to calculate Mean Absolute Change
        :return: (float) Mean Absolute Change
        """
        dif = np.abs(data[1:] - data[:-1])
        return np.divide(np.nansum(dif), data.shape[0] - np.sum(np.isnan(dif)))

    def approximate_entropy(self, data, window: int, level: float):
        """
        Calculates Approximate Entropy
        :param data: (numpy.ndarray) Data to calculate Approximate Entropy
        :param window: (int) Length of a run of data
        :param level: (float) Filtering level
        :return: (float) Approximate Entropy
        """
        def phi(window):
            win_ln = data.shape[0] - window + 1
            windows = np.vstack([data[i: i + window].reshape(1, -1) for i in range(win_ln)])
            count = np.array([sum([1 for w2 in windows if np.nanmax(np.abs(w1 - w2)) <= level]) / win_ln
                              for w1 in windows])
            return np.divide(np.sum(np.log(count)), win_ln)

        return phi(window) - phi(window + 1)


class StatsResults(Statistics):
    def mape(self, actual, forecast):
        """
        Calculates Mean Absolute Percentage Error given actual and predicted values
        :param actual: (np.ndarray) Actual values
        :param forecast: (np.ndarray) Predicted values
        :return: (float) MAPE score
        """
        return np.mean(np.abs(actual - forecast) / actual)

    def wmape(self, actual, forecast):
        """
        Calculates Weighted Mean Absolute Percentage Error given actual and predicted values
        :param actual: (np.ndarray) Actual values
        :param forecast: (np.ndarray) Predicted values
        :return: (float) MAPE score
        """
        return np.sum(np.abs(actual - forecast)) / np.sum(actual)

    def mae(self, actual, forecast):
        """
        Calculates Mean Absolute Error given actual and predicted values
        :param actual: (np.ndarray) Actual values
        :param forecast: (np.ndarray) Predicted values
        :return: (float) MAE score
        """
        return np.mean(np.abs(actual - forecast))

    def rmse(self, actual, forecast):
        """
        Calculates Root Mean Square Error given actual and predicted values
        :param actual: (np.ndarray) Actual values
        :param forecast: (np.ndarray) Predicted values
        :return: (float) RMSE score
        """
        return np.mean(np.square(actual - forecast)) ** 0.5

    def mse(self, actual, forecast):
        """
        Calculates Mean Square Error given actual and predicted values
        :param actual: (np.ndarray) Actual values
        :param forecast: (np.ndarray) Predicted values
        :return: (float) MSE score
        """
        return np.mean(np.square(actual - forecast))

    def r2(self, actual, forecast):
        """
        Calculates R-Squared given actual and predicted values
        :param actual: (np.ndarray) Actual values
        :param forecast: (np.ndarray) Predicted values
        :return: (float) R-Squared score
        """
        ln = len(actual)
        return (((ln * np.sum(np.multiply(actual, forecast)) - np.sum(actual) * np.sum(forecast)) /
                (((ln * np.sum(np.square(actual)) - np.sum(actual) ** 2) ** 0.5) * ((ln * np.sum(np.square(forecast)) -
                                                                                     np.sum(forecast) ** 2) ** 0.5)))
                ** 2)

    def box_pierce(self, resids, lags=[10]):
        """
        Box-Pierce portmanteau test
        :param resids: (numpy.ndarray) Residuals of the fitted model
        :param lags: (int or list(int)) Lags to return test values
        :return: (tuple(float)) Box-Pierce q-value and p-value
        """
        return tuple(acorr_ljungbox(resids, boxpierce=True, lags=[lags])[['bp_stat', 'bp_pvalue']].values[0])

    def ljung_box(self, resids, lags=[10]):
        """
        Ljung-Box portmanteau test
        :param resids: (numpy.ndarray) Residuals of the fitted model
        :param lags: (int or list(int)) Lags to return test values
        :return: (tuple(float)) Ljung-Box q-value and p-value
        """
        return tuple(acorr_ljungbox(resids, lags=[lags])[['lb_stat', 'lb_pvalue']].values[0])
