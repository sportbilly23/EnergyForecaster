from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from darts.models import TransformerModel


class Model:
    RandomForestRegressor = RandomForestRegressor
    SARIMAX = SARIMAX
    TransformerModel = TransformerModel

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.results = None

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
        if isinstance(self.model, RandomForestRegressor) and self.results:
            forecast_results = self.model.predict(data)
            forecast = forecast_results[start: start + steps]
            # TODO: confidence intervals
            # if alpha:
            #     return forecast, forecast_results.conf_int(alpha=alpha)[start: start + steps]
            return forecast
        if isinstance(self.model, TransformerModel):
            if isinstance(steps, type(None)):
                steps = self.model.output_chunk_length
            if start > self.model.output_chunk_length:
                start = 0
            if start + steps > self.model.output_chunk_length:
                steps = self.model.output_chunk_length - start
            return self.model.predict()[start: start + steps]

        raise NameError('Model type is not defined')

    def get_residuals(self):
        """
        Returns residuals of the model
        :return: (numpy.ndarray) residuals of the model
        """
        if isinstance(self.model, SARIMAX):
            return self.results.resid
        if isinstance(self.model, RandomForestRegressor):
            return self.results['resid']

    def fit(self, data, target):
        """
        Trains the model
        :param data: (numpy.ndarray) training dataset
        :param target: (numpy.ndarray) target dataset
        :return: (None)
        """
        if isinstance(self.model, SARIMAX):
            self.results = self.model.fit()
        elif isinstance(self.model, RandomForestRegressor):
            target = target.flatten()
            self.model.fit(data, target)
            self.results = {'resid': target - self.model.predict(data)}



