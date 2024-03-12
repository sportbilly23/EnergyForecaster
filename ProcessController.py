import utils
from DictNoDupl import DictNoDupl
import numpy as np
import pytz
import datetime


class Process:
    def __init__(self, name, target=DictNoDupl(), data=DictNoDupl(), scale=None, timezone=pytz.utc,
                 lags=1, black_lags=0, target_length=1, train=.6, validation=.2, test=.2, models=[], EF=None):
        self.name = name
        self.data = data
        self.target = target
        self.scale = scale
        self.train = train
        self.validation = validation
        self.test = test
        self.models = models
        # self.attributes = DictNoDupl()
        self.target_length = target_length
        self.timezone = timezone
        self.lags = (black_lags * target_length,) + tuple((np.arange(1, lags) + black_lags) * target_length)
        self.black_lags = tuple(np.arange(black_lags) * target_length)
        self._EF = EF

    def __eq__(self, other):
        try:
            assert self.name == other.name
            if self.data:
                for d in self.data:
                    assert utils.arrays_are_equal(self.data[d], other.data[d])
            if self.target:
                for t in self.target:
                    assert utils.arrays_are_equal(self.target[t], other.target[t])
            if self.data or self.target:
                assert utils.arrays_are_equal(self.scale, other.scale)
            assert self.train == other.train
            assert self.validation == other.validation
            assert self.test == other.test
            assert self.models == other.models
            assert self.target_length == other.target_length
            assert self.timezone == other.timezone
            assert self.lags == other.lags
            assert self.black_lags == other.black_lags
            assert self._EF.data_controller.path == other._EF.data_controller.path
            assert self._EF.data_controller.name == other._EF.data_controller.name
        except AssertionError:
            return True
        return False

    def get_model(self, name):
        """
        Get model from file by name
        :param name: (str) Name of the model
        :return: (dict) Dictionary with all model's details
        """
        return self._EF.data_controller._get_model(name)

    def _get_evaluation(self, name, func, data_part):
        """
        Creates evaluation statistics for all models
        :param name: (str) The name of the model
        :param func: (func) Function from Statistics.py to use evaluation (MAPE, MAE, RMSE, MSE, R-squared)
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) Evaluations for the models
        """
        data = self.get_data(data_part)
        actual = self.get_target(data_part).flatten()
        model = self.get_model(name)
        forecast = model['results'].forecast(exog=data, steps=len(data))

        return func(actual, forecast)

    def _get_model_evaluation(self, name, func):
        """
        Creates model evaluation statistics for all models
        :param func: (func) Function from Statistics.py to use evaluation (AIC, AICc, BIC)
        :return: (dict) Evaluations for the models
        """
        res = self.get_residuals(name)
        model = self.get_model(name)
        if model['interface'] == 'statsmodels':
            k_params = model['model'].k_params

        return func(res, k_params)

    def mape(self, name, data_part='train'):
        """
        Returns Mean Absolute Percentage Error evaluation for all models
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) MAPE evaluation for all models
        """
        return self._get_evaluation(name, self._EF.results_statistics.mape, data_part)

    def mae(self, name, data_part='train'):
        """
        Returns Mean Absolute Error evaluation for all models
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) MAE evaluation for all models
        """
        return self._get_evaluation(name, self._EF.results_statistics.mae, data_part)

    def rmse(self, name, data_part='train'):
        """
        Returns Root Mean Square Error evaluation for all models
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) RMSE evaluation for all models
        """
        return self._get_evaluation(name, self._EF.results_statistics.rmse, data_part)

    def mse(self, name, data_part='train'):
        """
        Returns Mean Square Error evaluation for all models
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) MSE evaluation for all models
        """
        return self._get_evaluation(name, self._EF.results_statistics.mse, data_part)

    def r2(self, name, data_part='train'):
        """
        Returns R-Squared evaluation for all models
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) R-squared evaluation for all models
        """
        return self._get_evaluation(name, self._EF.results_statistics.r2, data_part)

    def aic(self, name):
        """
        Returns Akaike's Information Criterion for all models
        :param name: (str) The name of the model
        :return: (dict) AIC evaluation for all models
        """
        return self._get_model_evaluation(name, self._EF.results_statistics.aic)

    def aicc(self, name):
        """
        Returns Akaike's Information Criterion corrected for all models
        :param name: (str) The name of the model
        :return: (dict) AICc evaluation for all models
        """
        return self._get_model_evaluation(name, self._EF.results_statistics.aicc)

    def bic(self, name):
        """
        Returns Bayesian Information Criterion
        :param name: (str) The name of the model
        :return: (dict) BIC evaluation for all models
        """
        return self._get_model_evaluation(name, self._EF.results_statistics.bic)

    def box_pierce(self, name, lags=[10]):
        """
        Box-Pierce portmanteau test
        :param name: (str) The name of the model
        :param lags: (int or list(int)) Lags to return test values
        :return: (tuple(float)) Box-Pierce q-value and p-value
        """
        return self._EF.results_statistics.box_pierce(self.get_residuals(name), lags)

    def ljung_box(self, name, lags=[10]):
        """
        Ljung-Box portmanteau test
        :param name: (str) The name of the model
        :param lags: (int or list(int)) Lags to return test values
        :return: (tuple(float)) Ljung-Box q-value and p-value
        """
        return self._EF.results_statistics.ljung_box(self.get_residuals(name), lags)

    def get_forecasts(self, name, data_part='train', start=0, steps=None, alpha=None):
        """
        Returns forecasts for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the forecasts ('train', 'validation', 'test')
        :param start: (int) Starting point at data
        :param steps: (int) Number of steps to forecast
        :return: (numpy.ndarray) Forecasts of the model
        """
        model = self.get_model(name)
        data = self.get_data(data_part)
        if not steps:
            steps = len(data)
        if model['interface'] == 'statsmodels' and 'results' in model:
            forecast_results = model['results'].get_forecast(exog=data, steps=len(data))
            forecast = forecast_results.predicted_mean[start: start + steps]
            if alpha:
                return forecast, forecast_results.conf_int(alpha=alpha)[start: start + steps]
            return forecast

    def get_residuals(self, name):
        """
        Returns residuals of a model by name
        :param name: (str) name of the model
        :return: (numpy.ndarray) residuals of the model
        """
        data = self.get_data()
        target = self.get_target().flatten()
        model = self._EF.data_controller._get_model(name)
        if model['interface'] == 'statsmodels' and 'results' in model:
            return target - model['results'].forecast(exog=data, steps=len(data))

    def get_all_residuals(self):
        """
        Returns residuals of all models
        :return: (dict) The residuals of all the models
        """
        resids = {}
        data = self.get_data()
        target = self.get_target().flatten()
        for name in self.models:
            model = self._EF.data_controller._get_model(name)
            if model['interface'] == 'statsmodels' and 'results' in model:
                resids.update({name: target - model['results'].forecast(exog=data, steps=len(data))})

        return resids

    def get_data(self, data_part='train'):
        """
        Returns the defined part of the dataset
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (numpy.ndarray) The defined part of the dataset
        """
        data = self._prepare_data(self.data)
        from_ = 0 if data_part == 'train' \
            else data.shape[0] * self.train if data_part == 'validation' \
            else data.shape[0] * (self.train + self.validation)
        to_ = data.shape[0] * self.train if data_part == 'train' \
            else data.shape[0] * (self.train + self.validation) if data_part == 'validation' \
            else data.shape[0]
        return data[int(from_):int(to_)]

    def get_target(self, data_part='train'):
        """
        Returns the defined part of the target dataset
        :param data_part: (str) The part of target data to get('train', 'validation', 'test')
        :return: (numpy.ndarray) The defined part of the target dataset
        """
        data = self._prepare_data(self.target)
        from_ = 0 if data_part == 'train' \
            else data.shape[0] * self.train if data_part == 'validation' \
            else data.shape[0] * (self.train + self.validation)
        to_ = data.shape[0] * self.train if data_part == 'train' \
            else data.shape[0] * (self.train + self.validation) if data_part == 'validation' \
            else data.shape[0]
        return data[int(from_):int(to_)]

    def get_scale(self, data_part='train'):
        """
        Returns the defined part of the scale
        :param data_part: (str) The part of target data to get('train', 'validation', 'test')
        :return: (numpy.ndarray) The defined part of the scale
        """
        from_ = 0 if data_part == 'train' \
            else self.scale.shape[0] * self.train if data_part == 'validation' \
            else self.scale.shape[0] * (self.train + self.validation)
        to_ = self.scale.shape[0] * self.train if data_part == 'train' \
            else self.scale.shape[0] * (self.train + self.validation) if data_part == 'validation' \
            else self.scale.shape[0]
        return self.scale[int(from_):int(to_)]

    def _prepare_data(self, data_dict):
        """
        Prepare data dictionary to feed a model
        :param data_dict: (DictNoDupl) Data in dictionary
        :return: (numpy.ndarray)
        """
        final_data = []
        for i in list(data_dict.keys()):
            names = data_dict[i].dtype.names
            if names:
                for name in names:
                    final_data.append(data_dict[i][name])
            else:
                final_data.append(data_dict[i])
        return np.vstack(final_data).T

    def plot_residuals(self, name, start=0, steps=None, axes=None):
        """
        Gets residuals of a model and scale and supplies results_visualizer create a plot
        :param name: (str) name of the model
        :param start: (int) starting point for the plot
        :param steps: (int) steps to depict on the plot
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        resids = self.get_residuals(name)
        ln = len(resids)
        scale = self.get_scale()[start: start + steps if steps else ln - start]
        scale_str = utils.timestamp_to_date_str(scale)
        self._EF.results_visualizer.plot_residuals(scale_str,
                                                   resids[start: start + steps if steps else ln - start],
                                                   f'{name} residuals', axes=axes)

    def hist_residuals(self, name, bins=10, density=False, plot_norm=False, axes=None):
        """
        Gets residuals of a model and supplies results_visualizer to create a histogram plot
        :param name: (str) name of the model
        :param bins: (int) number of histogram bins
        :param density: (bool) If True it creates probability density histogram
        :param plot_norm: (bool) If True and also density is True, it draws a normal distribution with same mean and
                                 standard deviation as these of the data.
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        resids = self.get_residuals(name)
        self._EF.results_visualizer.hist(resids, f'{name} residuals', bins=bins, density=density,
                                         plot_norm=plot_norm, axes=axes)

    def plot_forecast(self, name, data_part='train', start=0, steps=None, alpha=None, axes=None,
                      intervals_from_residuals=True):
        """
        Gets forecasts of a model and supplies results_visualizer to create a plot
        :param name: (str) name of the model
        :param data_part: (str) The part of target data to get('train', 'validation', 'test')
        :param start: (int) starting point for the plot
        :param steps: (int) steps to depict on the plot
        :param alpha: (float) alpha for prediction intervals
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :param intervals_from_residuals: (bool) True to create prediction interval from residuals distribution
        :return: (pyplot.axes) axes of the plot
        """
        alpha = min(alpha, 1 - alpha)
        forecast = self.get_forecasts(name, data_part=data_part, start=start, steps=steps,
                                      alpha=None if intervals_from_residuals else alpha)
        conf_int = []
        if alpha:
            if intervals_from_residuals:
                from scipy.stats import norm
                resids = self.get_residuals(name)
                mn = np.mean(resids)
                std = np.std(resids)
                conf_int = [(i + norm.ppf(alpha, mn, std), i + norm.ppf(1 - alpha, mn, std)) for i in forecast]
            else:
                forecast, conf_int = forecast

        actual = self.get_target(data_part)[start: start + len(forecast)]
        scale = utils.timestamp_to_date_str(self.get_scale(data_part)[start: start + len(forecast)], self.timezone)
        self._EF.results_visualizer.plot_forecast(scale, actual, forecast, conf_int,
                                                  name=f'{name}' + ('' if isinstance(alpha, type(None)) else
                                                  f' - confidence {1 - alpha:1.0%}'), axes=axes)


class ProcessController:
    """
    Manages processes for Energy Forecaster
    """

    def __init__(self, ef=None):
        self._EF = ef
        self.process = None

    def set_model(self, model, name, interface):
        """
        Insert model to the current process
        :param model: A well-defined model
        :param name: (str) A name for the model
        :param interface: (str) The interface to control the model training and the
        :return:
        """
        self._EF.data_controller._set_model(name, model, interface)
        self.process.models.append(name)

    def _process_creation_from_file(self, name, target, data, scale, timezone, lags, black_lags, target_length, train,
                                    validation, test, models):
        """
        Create back a saved process
        :param name: (str) Name of the process
        :param target: (DictNoDupl) Target data
        :param data: (DictNoDupl) Data regressors
        :param scale: (numpy.ndarray) Scale of the data
        :param timezone: (pytz.timezone) Timezone of the scale
        :param lags: (tuple) Tuple of lags used in process
        :param black_lags: (tuple) Tuple of black-lags used in process
        :param target_length: Length of target
        :param train: (float) Proportion for training data
        :param validation: (float) Proportion for data validation
        :param test: (float) Proportion for test data
        :param models: (list(str)) List of files used as model definition-training-results storage
        :return: (Process) The saved process
        """
        return Process(name, target, data, scale, timezone, len(lags), len(black_lags),
                       target_length, train, validation, test, models, self._EF)

    def fit_models(self):
        """
        Training all models
        :return: (None)
        """
        for name in self.process.models:
            model = self._EF.data_controller._get_model(name)
            if 'results' not in model:
                if model['interface'] == 'statsmodels':
                    model.update({'results': model['model'].fit()})
                    self._EF.data_controller._update_model(model)

    def set_process(self, name, lags=0, black_lags=0, target_length=1, update_file=False, train=.6, validation=.2, test=.2):
        """
        Defines a new process
        :param name: (str) Name of the process
        :param lags: (int) Number of historical data lags
        :param black_lags: (int) Number of black historical data lags
        :param target_length: (int) Length of target
        :param update_file: (bool) True to update file
        :param train: (float) Proportion for training data
        :param validation: (float) Proportion for data validation
        :param test: (float) Proportion for test data
        :return: (None)
        """
        amount = train + validation + test
        self._EF.data_controller.set_process(Process(name, lags=lags, black_lags=black_lags,
                                                     target_length=target_length, train=train / amount,
                                                     validation=validation / amount, test=test / amount,
                                                     EF=self._EF), update_file=update_file)

    def get_process_names(self):
        """
        Get all names of the processes
        :return: (list(str)) List of names
        """
        return self._EF.data_controller.get_process_names()

    def get_process(self, name):
        """
        Get process with given name
        :param name: (str) Name of the process
        :return: (Process) The process
        """
        self.process = self._EF.data_controller.get_process(name)

    def close_process(self):
        """
        Removes a process from memory
        :return: (None)
        """
        if self.is_process_changed():
            yn = input("Process has been changed. Are you sure you want to close it? (Type 'yes' to close): ")
            if yn.upper() != 'YES':
                return
        self.process = None

    def update_process(self):
        """
        Stores the updated current process to the file
        :return: (None)
        """
        if self.process:
            self._EF.data_controller.update_process(self.process)
        else:
            raise ValueError('No process selected')

    def remove_process(self, name):
        """
        Removes a process from the file
        :param name: (str) Name of the process
        :return: (None)
        """
        yn = input("Process will de deleted permanently. Are you sure you want to delete it? (Type 'yes' to delete): ")
        if yn.upper() == 'YES':
            if self.process and name == self.process.name:
                self.process = None
            self._EF.data_controller.remove_process(name)

    def is_process_changed(self):
        """
        Checks if current process has been changed
        :return: (bool) True if current process have changes between RAM and file.
        """
        return self._EF.data_controller.is_process_changed(self.process)

    def insert_data(self, dataset, columns, change_to_new_tzone=True, no_lags=True):
        """
        Addind data to the process
        :param dataset: (str) Name of the dataset
        :param columns: (list(str)) Names of the columns to be added
        :param change_to_new_tzone: (bool) If True, it changes the timezone with respect of new data timezone
        :return: (None)
        """
        if self.process:
            self._EF.data_controller._import_data_to_process(dataset, columns, self.process,
                                                             change_to_new_tzone=change_to_new_tzone, no_lags=no_lags)


