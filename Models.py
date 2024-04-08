import os
import tempfile
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from darts.models import TransformerModel
from darts.timeseries import TimeSeries
from xarray import DataArray
from pandas import DatetimeIndex
import numpy as np


class Model:
    RandomForestRegressor = RandomForestRegressor
    SARIMAX = SARIMAX
    TransformerModel = TransformerModel

    def __init__(self, name, model, fit_params={}):
        self.name = name
        self.model = model
        self.results = None
        self.fit_params = fit_params

    def aic(self):
        """
        Calculates Akaike's Information Criterion of residuals
        :return: (float) AIC score
        """
        if isinstance(self.model, SARIMAX):
            return self.model.aic
        if isinstance(self.model, RandomForestRegressor):
            return None
        if isinstance(self.model, TransformerModel):
            return None

    def aicc(self):
        """
        Calculates Akaike's Information Criterion corrected of residuals
        :return: (float) AICc score
        """
        if isinstance(self.model, SARIMAX):
            return self.model.aicc
        if isinstance(self.model, RandomForestRegressor):
            return None
        if isinstance(self.model, TransformerModel):
            return None

    def bic(self):
        """
        Calculates Bayesian Information Criterion of residuals
        :return: (float) BIC score
        """
        if isinstance(self.model, SARIMAX):
            return self.model.bic
        if isinstance(self.model, RandomForestRegressor):
            return None
        if isinstance(self.model, TransformerModel):
            return None

    def get_forecasts(self, data, start=0, steps=None, alpha=None):
        """
        Returns forecasts
        :param data: (numpy.ndarray) new data for making forecasts
        :param start: (int) starting point at data
        :param steps: (int) number of steps to forecast
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :return:
        """
        if isinstance(self.model, SARIMAX) and self.results:
            forecast_results = self.results.get_forecast(exog=data, steps=len(data))
            forecast = forecast_results.predicted_mean[start: start + steps]
            if alpha:
                return forecast, forecast_results.conf_int(alpha=alpha)[start: start + steps]
            return forecast
        elif isinstance(self.model, RandomForestRegressor) and self.results:
            forecast_results = self.model.predict(data)
            forecast = forecast_results[start: start + steps]
            # TODO: confidence intervals
            # if alpha:
            #     return forecast, forecast_results.conf_int(alpha=alpha)[start: start + steps]
            return forecast
        elif self.model == TransformerModel:
            model = self._open_darts_model()
            if isinstance(steps, type(None)):
                steps = model.output_chunk_length
            if start > model.output_chunk_length:
                start = 0
            if start + steps > model.output_chunk_length:
                steps = model.output_chunk_length - start
            preds = model.predict(model.output_chunk_length)

            return preds[start: start + steps].all_values().flatten(), (start, steps)

        raise NameError('Model type is not defined')

    def get_residuals(self):
        """
        Returns residuals of the model
        :return: (numpy.ndarray) residuals of the model
        """
        if isinstance(self.model, SARIMAX):
            return self.results.resid
        elif isinstance(self.model, RandomForestRegressor):
            return self.results['resid']
        elif isinstance(self.model, TransformerModel):
            return None

    def fit(self, data, target=None, scale=None):
        """
        Trains the model
        :param data: (numpy.ndarray) training dataset
        :param target: (numpy.ndarray) target dataset
        :return: (None)
        """
        if isinstance(self.model, SARIMAX):
            self.results = self.model.fit(**self.fit_params)
        elif isinstance(self.model, RandomForestRegressor):
            target = target.flatten()
            self.model.fit(data, target, **self.fit_params)
            self.results = {'resid': target - self.model.predict(data)}
        elif isinstance(self.model, TransformerModel):
            data = np.expand_dims(data, 2)
            target = np.expand_dims(target, 2)
            scale = DatetimeIndex(scale * 1000000000)

            series = DataArray(target, dims=['scale', 'component', 'sample'],
                               coords=dict(scale=scale),
                               attrs=dict(static_covariates=None, hierarchy=None))
            series = TimeSeries(series)

            past_cov = DataArray(data, dims=['scale', 'component', 'sample'],
                                 coords=dict(scale=scale),
                                 attrs=dict(static_covariates=None, hierarchy=None))
            past_cov = TimeSeries(past_cov)

            self.model.fit(series, past_cov)
            self.results = {}

            temp_file = self._temp_file_and_cpkt()
            self.model.save(temp_file)
            with open(temp_file, 'rb') as f:
                self.results.update({'model': f.read()})
            with open(f'{temp_file}.ckpt', 'rb') as f:
                self.results.update({'weights': f.read()})
            os.remove(temp_file)
            os.remove(f'{temp_file}.ckpt')
            self.model = TransformerModel

    def _open_darts_model(self):
        temp_file = tempfile.mktemp()
        with open(temp_file, 'wb') as f:
            f.write(self.results['model'])
        with open(f'{temp_file}.ckpt', 'wb') as f:
            f.write(self.results['weights'])
        trained_model = self.model.load(temp_file)
        os.remove(temp_file)
        os.remove(f'{temp_file}.ckpt')

        return trained_model

    def _temp_file_and_cpkt(self):
        while True:
            temp_file = tempfile.mktemp()
            if not os.path.isfile(f'{temp_file}.ckpt'):
                break
        return temp_file
