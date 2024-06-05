import numpy as np

from DictNoDupl import DictNoDupl
import pytz
from Models import *


CMDS = {'process': ['insert_data'],
        'process_controller': ['set_process', 'get_process', 'update_process'],
        'data_controller': ['import_csv', 'get_dataset', 'update_dataset'],
        'preprocessor': ['log', 'log2', 'log10', 'ln', 'exp', 'exp2', 'boxcox', 'limit_output', 'minmax', 'standard',
                         'robust', 'differentiate', 'croston_method', 'to_timestamp', 'weekend', 'weekday', 'monthday',
                         'day_hour', 'year_day', 'year_week', 'year_month', 'month_weekdays', 'public_holidays',
                         'lagged_series', 'fill_backward', 'fill_forward', 'fill_linear', 'downgrade_data_frequency',
                         'attach_scale', 'make_target']}


class Process:
    def __init__(self, name, target=DictNoDupl(), data=DictNoDupl(), scale=None, data_index={}, target_index={},
                 timezone=pytz.utc, lags=1, black_lags=0, measure_period=1, train=.6, validation=.2, test=.2, models=[],
                 attributes=DictNoDupl(), EF=None):
        self.name = name
        self.data = data
        self.target = target
        self.scale = scale
        self.data_index = data_index
        self.target_index = target_index
        self.train = train
        self.validation = validation
        self.test = test
        self.models = models
        self.attributes = attributes
        self.measure_period = measure_period
        self.timezone = timezone
        self.lags = lags
        self.black_lags = black_lags
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
            assert self.data_index == other.data_index
            assert self.target_index == other.target_index
            assert self.train == other.train
            assert self.validation == other.validation
            assert self.test == other.test
            assert self.models == other.models
            assert self.attributes == other.attributes
            assert self.measure_period == other.measure_period
            assert self.timezone == other.timezone
            assert self.lags == other.lags
            assert self.black_lags == other.black_lags
            assert self._EF.data_controller.path == other._EF.data_controller.path
            assert self._EF.data_controller.name == other._EF.data_controller.name
        except AssertionError:
            return True
        return False

    def get_units(self, column):
        """
        Returns the units of a column with respect of transformations
        :param column: (str) Label of the column
        :return: (str) Units of a column with respect of transformations
        """
        units = self.attributes[column]['units']
        for trans_dict in self.attributes[column]['transformations']:
            trans = str(trans_dict['func'])
            units = f"{trans[trans.index('Preprocessor.') + 13: trans.index('<locals>') - 1]}({units})"
        return units

    def reverse_trans(self, column):
        """
        Returns data applying reverse transformation functions
        :param column: (str) Label of the column
        :return: (numpy.ndarray) Initial data of a column (before transformations)
        """
        source = self.target if column in self.target else self.data
        return self._EF.preprocessor.reverse_trans(source[column], self.attributes[column]['transformations'])

    def get_model(self, name):
        """
        Get model from file by name
        :param name: (str) Name of the model
        :return: (dict) Dictionary with all model's details
        """
        return self._EF.data_controller._get_model(name)

    def _get_evaluation(self, name, func, data_part, torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Creates evaluation statistics for a model
        :param name: (str) The name of the model
        :param func: (func) Function from Statistics.py to use evaluation (MAPE, MAE, RMSE, MSE, R-squared)
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) Evaluations for the models
        """
        actual = self.get_target(data_part).flatten()
        forecast = self.get_forecasts(name, data_part, torch_best_valid=torch_best_valid,
                                      torch_best_loss_if_no_valid=torch_best_loss_if_no_valid).flatten()

        if isinstance(forecast, dict):
            forecast = forecast['forecast']

        return func(actual[:len(forecast)], forecast)

    def mape(self, name, data_part='train', torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns Mean Absolute Percentage Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) MAPE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.mape, data_part,
                                    torch_best_valid=torch_best_valid,
                                    torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def wmape(self, name, data_part='train', torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns weighted Mean Absolute Percentage Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) wMAPE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.wmape, data_part,
                                    torch_best_valid=torch_best_valid,
                                    torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def mae(self, name, data_part='train', torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns Mean Absolute Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) MAE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.mae, data_part, torch_best_valid=torch_best_valid,
                                    torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def rmse(self, name, data_part='train', torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns Root Mean Square Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) RMSE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.rmse, data_part,
                                    torch_best_valid=torch_best_valid,
                                    torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def mse(self, name, data_part='train', torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns Mean Square Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) MSE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.mse, data_part, torch_best_valid=torch_best_valid,
                                    torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def r2(self, name, data_part='train', torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns R-Squared evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) R-squared evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.r2, data_part, torch_best_valid=torch_best_valid,
                                    torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def aic(self, name):
        """
        Returns Akaike's Information Criterion for a model
        :param name: (str) The name of the model
        :return: (float) AIC evaluation for a model
        """
        model = self.get_model(name)
        return model.aic

    def aicc(self, name):
        """
        Returns Akaike's Information Criterion corrected for a model
        :param name: (str) The name of the model
        :return: (float) AICc evaluation for a model
        """
        model = self.get_model(name)
        return model.aicc

    def bic(self, name):
        """
        Returns Bayesian Information Criterion for a model
        :param name: (str) The name of the model
        :return: (float) BIC evaluation for a model
        """
        model = self.get_model(name)
        return model.bic

    def box_pierce(self, name, lags=[10], torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Box-Pierce portmanteau test for a model
        :param name: (str) The name of the model
        :param lags: (int or list(int)) Lags to return test values
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (tuple(float)) Box-Pierce q-value and p-value
        """
        return self._EF.results_statistics.box_pierce(self.get_residuals(name, torch_best_valid,
                                                                         torch_best_loss_if_no_valid), lags)

    def ljung_box(self, name, lags=[10], torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Ljung-Box portmanteau test for a model
        :param name: (str) The name of the model
        :param lags: (int or list(int)) Lags to return test values
        :param lags: (int or list(int)) Lags to return test values
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (tuple(float)) Ljung-Box q-value and p-value
        """
        return self._EF.results_statistics.ljung_box(self.get_residuals(name, torch_best_valid,
                                                                        torch_best_loss_if_no_valid), lags)

    def get_forecasts(self, name, data_part='train', start=0, steps=None, alpha=None, torch_best_valid=True,
                      torch_best_loss_if_no_valid=True, intervals_from_validation=True):
        """
        Returns forecasts for a model
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the forecasts ('train', 'validation', 'test')
        :param start: (int) Starting point at data
        :param steps: (int) Number of steps to forecast
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :param intervals_from_validation: (bool) True to calculate intervals from validation data
        :return: (numpy.ndarray) Forecasts of the model
        """
        model = self.get_model(name)
        if intervals_from_validation and isinstance(model.results, dict) and 'valid_resid' not in model.results:
            if isinstance(model, VotingModel):
                for fn in model.model_filenames:
                    with open(fn, 'rb') as f:
                        md = dill.load(f)
                        _ = md.get_validation_residuals(data=self.get_data('validation'),
                                                        target=self.get_target('validation'))
                        self._EF.data_controller._update_model(md)
            _ = model.get_validation_residuals(data=self.get_data('validation'), target=self.get_target('validation'))
            self._EF.data_controller._update_model(model)
        data = self.get_data(data_part)
        return self._get_forecasts(model, data, start, steps, alpha, torch_best_valid=torch_best_valid,
                                   torch_best_loss_if_no_valid=torch_best_loss_if_no_valid,
                                   intervals_from_validation=intervals_from_validation)

    def _get_forecasts(self, model, data, start=0, steps=None, alpha=None, torch_best_valid=True,
                       torch_best_loss_if_no_valid=True, intervals_from_validation=True):
        """
        Returns forecasts for a model
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :param model: (str) the model
        :param data: (str) the data for the forecasts
        :param start: (int) Starting point at data
        :param steps: (int) Number of steps to forecast
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :param intervals_from_validation: (bool) True to calculate intervals from validation data
        :return: (numpy.ndarray) Forecasts of the model
        """
        return model.get_forecasts(data, start, steps, alpha, torch_best_valid=torch_best_valid,
                                   torch_best_loss_if_no_valid=torch_best_loss_if_no_valid,
                                   intervals_from_validation=intervals_from_validation)

    def get_residuals(self, name, torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns residuals of a model by name
        :param name: (str) name of the model
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (numpy.ndarray) residuals of the model
        """
        model = self.get_model(name)
        return model.get_residuals(torch_best_valid=torch_best_valid,
                                   torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

    def get_validation_residuals(self, name):
        """
        Returns validation residuals of a model by name
        :param name: (str) name of the model
        :return: (numpy.ndarray) residuals of the model
        """
        model = self.get_model(name)
        if isinstance(model.results, dict) and 'valid_resid' in model.results:
            return model.results['valid_resid']
        resids = model.get_validation_residuals(data=self.get_data('validation'), target=self.get_target('validation'))
        self._EF.data_controller._update_model(model)
        return resids

    def get_data(self, data_part='train'):
        """
        Returns the defined part of the dataset
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (numpy.ndarray) The defined part of the dataset
        """
        data = self._prepare_data(self.data, self.data_index)
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
        data = self._prepare_data(self.target, self.target_index)
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

    def _prepare_data(self, data_dict, data_index):
        """
        Prepare data dictionary to feed a model
        :param data_dict: (DictNoDupl) Data in dictionary
        :return: (numpy.ndarray)
        """
        final_data = []
        for i in range(len(data_dict)):
            names = data_dict[data_index[i]].dtype.names
            if names:
                for name in names:
                    final_data.append(data_dict[data_index[i]][name])
            else:
                final_data.append(data_dict[data_index[i]])
        return np.vstack(final_data).T

    def plot_residuals(self, name, start=0, steps=None, axes=None, torch_best_valid=True,
                       torch_best_loss_if_no_valid=True):
        """
        Gets residuals of a model and scale and supplies results_visualizer create a plot
        :param name: (str) name of the model
        :param start: (int) starting point for the plot
        :param steps: (int) steps to depict on the plot
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (pyplot.axes) axes of the plot
        """
        resids = self.get_residuals(name, torch_best_valid, torch_best_loss_if_no_valid)
        ln = len(resids)
        scale = self.get_scale()[start: start + steps if steps else ln - start]
        scale_str = utils.timestamp_to_date_str(scale, self.timezone)
        self._EF.results_visualizer.plot_residuals(scale_str,
                                                   resids[start: start + steps if steps else ln - start],
                                                   f'{name} residuals', axes=axes)

    def hist_residuals(self, name, bins=10, density=False, plot_norm=False, axes=None, torch_best_valid=True,
                       torch_best_loss_if_no_valid=True):
        """
        Gets residuals of a model and supplies results_visualizer to create a histogram plot
        :param name: (str) name of the model
        :param bins: (int) number of histogram bins
        :param density: (bool) If True it creates probability density histogram
        :param plot_norm: (bool) If True and also density is True, it draws a normal distribution with same mean and
                                 standard deviation as these of the data.
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (pyplot.axes) axes of the plot
        """
        resids = self.get_residuals(name, torch_best_valid, torch_best_loss_if_no_valid)
        self._EF.results_visualizer.hist(resids, f'{name} residuals', bins=bins, density=density,
                                         plot_norm=plot_norm, axes=axes)

    def add_model(self, name):
        """
        Add an existed model to the process
        :param name: (str) name of the model
        :return: (None)
        """
        if name in self.models:
            raise KeyError('Model is already registered in the process')
        elif name not in self._EF.data_controller.get_model_names():
            raise KeyError('Model name not exists')
        else:
            self.models.append(name)

    def remove_model(self, name):
        """
        Remove a registered model from the process
        :param name: (str) name of the model
        :return: (None)
        """
        if name not in self.models:
            raise KeyError('Model name not registered in the process')
        else:
            yn = input("Model will de removed from the process. Are you sure? (Type 'yes' to delete): ")
            if yn.upper() == 'YES':
                self.models.remove(name)

    def plot_forecasts(self, name, data_part='train', start=0, steps=None, alpha=None, axes=None,
                      intervals_from_validation=True, torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Gets forecasts of a model and supplies results_visualizer to create a plot
        :param name: (str) name of the model
        :param data_part: (str) The part of target data to get('train', 'validation', 'test')
        :param start: (int) starting point for the plot
        :param steps: (int) steps to depict on the plot
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :param intervals_from_validation: (bool) True to calculate intervals from validation data
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (pyplot.axes) axes of the plot
        """
        not_alpha = isinstance(alpha, type(None))
        if not not_alpha:
            alpha = min(alpha, 1 - alpha)
        if not_alpha or 0 < alpha <= .5:
            if intervals_from_validation:
                _ = self.get_validation_residuals(name)
            forecasts = self.get_forecasts(name, data_part, start, steps, alpha, torch_best_valid,
                                           torch_best_loss_if_no_valid, intervals_from_validation)

            forecast = forecasts['forecast'].flatten()

            actual = self.get_target(data_part)[forecasts['start']: (forecasts['start'] + forecasts['steps'])\
                                                if forecasts['steps'] else forecasts['steps']].flatten()
            scale = utils.timestamp_to_date_str(self.get_scale(data_part)[forecasts['start']: forecasts['start'] +
                                                                          forecasts['steps']], self.timezone)
            self._EF.results_visualizer.plot_forecasts(scale, actual, forecast, forecasts['conf_int'],
                                                      name=f'{name}' + ('' if isinstance(alpha, type(None)) else
                                                                        f' - confidence {1 - alpha: 1.0%}'), axes=axes)
        else:
            raise ValueError('Alpha must be a float number between 0 and 1')

    def plot_shapes(self, columns, data_part='train', axes=None):
        """
        Compares plot-shapes of different variables
        :param columns: (list(str)) columns of the variables
        :param data_part: (str) The part of target data to get('train', 'validation', 'test')
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        scale = utils.timestamp_to_date_str(self.get_scale(data_part), self.timezone)

        datas = [self.get_target(data_part)[:, [i for i in self.target_index
                                                if self.target_index[i] == column][0]] if column in self.target else
                 self.get_data(data_part)[:, [i for i in self.data_index
                                              if self.data_index[i] == column][0]] if column in self.data else
                 None for column in columns]
        check_nones = [isinstance(i, type(None)) for i in datas]
        try:
            error = check_nones.index(None)
            raise KeyError(f'Column {check_nones[error]} not exists')
        except ValueError:
            return self._EF.data_visualizer.plot_shapes(scale, datas, columns, axes=axes)

    def plot_loss_by_epoch(self, name, axes=None):
        model = self.get_model(name)
        if isinstance(model.model, TorchModel):
            self._EF.results_visualizer.plot_loss_by_epoch(model.model.loss_history,
                                                           name=f'{name} - loss progress', axes=axes)

    def plot_validation_by_epoch(self, name, axes=None):
        model = self.get_model(name)
        if isinstance(model.model, TorchModel) and not isinstance(model.model.validation_history, type(None)):
            self._EF.results_visualizer.plot_validation_by_epoch(model.model.validation_history,
                                                                 name=f'{name} - validation progress', axes=axes)

    def plot_loss_by_time(self, name, axes=None):
        model = self.get_model(name)
        if isinstance(model.model, TorchModel):
            self._EF.results_visualizer.plot_loss_by_time(model.model.epoch_times, model.model.loss_history,
                                                          name=f'{name} - loss progress by time', axes=axes)

    def plot_validation_by_time(self, name, axes=None):
        model = self.get_model(name)
        if isinstance(model.model, TorchModel) and not isinstance(model.model.validation_history, type(None)):
            self._EF.results_visualizer.plot_validation_by_time(model.model.epoch_times, model.model.validation_history,
                                                                name=f'{name} - validation progress by time', axes=axes)

    def plot_compare_models_loss(self, names, time=False, use_validation=False, axes=None):
        """
        Compares loss or validation of different models by epoch or by time
        :param names: (list(str)) names of models to compare
        :param time: (bool) True to compare by time and False to compare by epochs
        :param use_validation: (bool) True to use validation function or False to use loss function
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        losses = []
        times = []
        loss_func = None
        units = 'loss' if not use_validation else 'validation'
        for name in names:
            model = self.get_model(name)
            if isinstance(loss_func, type(None)):
                loss_func = model.model.loss_func.__class__.__name__ if not use_validation else model.model.validation_func
            else:
                if (not use_validation and loss_func != model.model.loss_func.__class__.__name__) or \
                        (use_validation and loss_func != model.model.validation_func.__class__.__name__):
                    raise ValueError(f'models must have same {units} function')
            losses.append(model.model.loss_history if not use_validation else model.model.validation_history)
            if time:
                times.append(np.cumsum(model.model.epoch_times))

        return self._EF.results_visualizer.plot_compare_models_loss(losses, names, times=times if time else None,
                                                                    units=units, axes=axes)

    def data_summary(self, columns=None):
        """
        Returns a summary for the data
        :param columns: (list(str)) list of the column names
        :return: (str) printable summary of data
        """
        return self._EF.data_controller.data_summary(self, columns)

    def fit_models(self, n_epochs=1, use_torch_validation=False):
        """
        Training all models
        :return: (None)
        """
        for name in self.models:
            self.fit_model(name, n_epochs, use_torch_validation)

    def fit_model(self, name, n_epochs=1, use_torch_validation=False):
        """
        Trains a model
        :param name: (str) name of the model to train
        :param n_epochs: (int) number of epochs to fit (if the model need it)
        :param use_torch_validation: (bool) evaluates model with validation data while training (for TorchModel only)
        :return: (None)
        """
        model = self.get_model(name)
        if isinstance(model, Model) and isinstance(model.results, type(None)):
            print(f'Fitting model "{name}"...')
            model.fit(self.get_data(), self.get_target(), scale=self.get_scale(), n_epochs=n_epochs,
                      validation_data=self.get_data('validation') if use_torch_validation else None,
                      validation_target=self.get_target('validation') if use_torch_validation else None)
            self._EF.data_controller._update_model(model)

    def extend_fit(self, name, n_epochs=1, use_torch_validation=False):
        """
        Extra fit for a model
        :param name: (str) name of the model
        :param n_epochs: (int) number of epochs
        :param use_torch_validation: (bool) evaluates model with validation data while training (for TorchModel only)
        :return: (None)
        """
        model = self.get_model(name)
        validation_data = None
        validation_target = None
        if isinstance(model.model, TorchModel):
            if model.model.validation_func:
                validation_data = self.get_data('validation')
                validation_target = self.get_target('validation')
        model.extend_fit(self.get_data(), self.get_target(), n_epochs,
                         validation_data=validation_data, validation_target=validation_target)
        self._EF.data_controller._update_model(model)

    def is_changed(self):
        """
        Checks if current process has been changed
        :return: (bool) True if current process have changes between RAM and file.
        """
        return self._EF.data_controller.is_process_changed(self)

    def insert_data(self, dataset: str, columns: [str], change_to_new_tzone: bool = True, no_lags: bool = True):
        """
        Addind data to the process
        :param no_lags: (bool) False to create data lags
        :param dataset: (str) Name of the dataset
        :param columns: (list(str)) Names of the columns to be added
        :param change_to_new_tzone: (bool) If True, it changes the timezone with respect of new data timezone
        :return: (None)
        """
        self._EF.data_controller._import_data_to_process(dataset, columns, self,
                                                         change_to_new_tzone=change_to_new_tzone, no_lags=no_lags)

    def evaluation_summary(self, data_part='train', names=None, torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns a summary of evaluations for all models of the process
        :param names: (list(str)) model names for evaluation. Select None to evaluate all models of the process
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (str) summary of evaluations for all models
        """
        models = self.models if isinstance(names, type(None)) else [n for n in names if n in self.models]
        ln = max([len(m) for m in models])
        evals = ['MAPE', 'wMAPE', 'MAE', 'RMSE', 'MSE', 'R2']
        data = self.get_data(data_part)
        evaluations = {e: {'left_digs': 1, 'values': {}} for e in evals}
        actuals = self.get_target(data_part).flatten()
        for m in models:
            model = self.get_model(m)
            forecasts = self._get_forecasts(model, data, torch_best_valid=torch_best_valid,
                                            torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

            for eval in evals:
                value = self._EF.results_statistics.__getattribute__(eval.lower())(actuals[:forecasts['steps']],
                                                                                   forecasts['forecast'].flatten())
                strings = f'{value:f}'.split('.')
                evaluations[eval]['left_digs'] = max(evaluations[eval]['left_digs'], len(strings[0].lstrip('0')))
                evaluations[eval]['values'].update({m: value})

        spaces = {e: max(7, evaluations[e]["left_digs"] + 3) for e in evals}
        decims = {e: max(2, 6 - evaluations[e]["left_digs"]) for e in evals}
        summary = [' | '.join([' ' * ln] + [f'{e}{" " * (spaces[e] - len(e))}' for e in evals])]
        summary.append(''.join(['|' if i == '|' else '-' for i in summary[0]]))
        for m in models:
            summary.append(' | '.join(
                [f'{m:<{ln}}'] + [f"{evaluations[e]['values'][m]:>{spaces[e]}.{decims[e]}f}"
                                  for e in evals]))

        return '\n'.join(summary)


class ProcessController:
    """
    Manages processes for Energy Forecaster
    """

    def __init__(self, ef=None):
        self._EF = ef
        self.process = None

    def set_model(self, model, name, fit_params={}, add_to_process=False):
        """
        Insert model to the Energy Forecaster
        :param model: a well-defined model
        :param name: (str) name of the model
        :param fit_params: (dict) dictionary with training parameters
        :param add_to_process: (bool) True to register model in current process
        :return: (None)
        """
        self._EF.data_controller._set_model(name, model, fit_params)
        if add_to_process:
            self.process.add_model(name)

    def set_voting_model(self, name, model_names, add_to_process=False):
        """
        Insert voting model to the Energy Forecaster
        :param name: (str) name of the voting model
        :param model_names: (list(str))
        :param add_to_process: (bool) True to register model in current process
        :return: (None)
        """
        self._EF.data_controller._set_voting_model(name, model_names)
        if add_to_process:
            self.process.add_model(name)

    def _process_creation_from_file(self, name, target, data, scale, data_index, target_index, timezone, lags,
                                    black_lags, measure_period, train, validation, test, models, attributes):
        """
        Creates instance of a saved process
        :param name: (str) Name of the process
        :param target: (DictNoDupl) Target data
        :param data: (DictNoDupl) Data regressors
        :param scale: (numpy.ndarray) Scale of the data
        :param data_index: (dict) Dictionary of data columns
        :param target_index: (dict) Dictionary of target columns
        :param timezone: (pytz.timezone) Timezone of the scale
        :param lags: (tuple) Tuple of lags used in process
        :param black_lags: (tuple) Tuple of black-lags used in process
        :param measure_period: number of instances between two set of measurements
        :param train: (float) Proportion for training data
        :param validation: (float) Proportion for data validation
        :param test: (float) Proportion for test data
        :param models: (list(str)) List of files used as model definition-training-results storage
        :param attributes: (dict) Dictionary of data and target attributes
        :return: (Process) The saved process
        """
        return Process(name, target, data, scale, data_index, target_index, timezone, lags, black_lags,
                       measure_period, train, validation, test, models, attributes, self._EF)

    def set_process(self, name: str, lags: int = 0, black_lags: int = 0, measure_period: int = 1,
                    update_file: bool = False, train: float = .6, validation: float = .2, test: float = .2):
        """
        Defines a new process
        :param name: (str) Name of the process
        :param lags: (int) Number of historical data lags
        :param black_lags: (int) Number of black historical data lags
        :param measure_period: (int) Number of instances between two set of measurements
        :param update_file: (bool) True to update file
        :param train: (float) Proportion for training data
        :param validation: (float) Proportion for data validation
        :param test: (float) Proportion for test data
        :return: (None)
        """
        amount = train + validation + test
        self._EF.data_controller.set_process(Process(name, lags=lags, black_lags=black_lags,
                                                     measure_period=measure_period, train=train / amount,
                                                     validation=validation / amount, test=test / amount,
                                                     EF=self._EF), update_file=update_file)

    def get_process_names(self):
        """
        Get all names of the processes
        :return: (list(str)) List of names
        """
        return self._EF.data_controller.get_process_names()

    def get_process(self, name: str):
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
        if self.process and self.process.is_changed():
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

    def run_process_script(self, filename):
        """
        Runs a script to automate data preprocessing
        :param filename: (str) filename of the script
        :return: (None)
        """
        def cast_value(annotation, value):
            return (True if value == 'True' else False) if annotation == bool else annotation(value)

        dataset = ''
        with open(filename, 'r') as f:
            script = f.readlines()
        for num, line in enumerate(script):
            if line != '\n':
                try:
                    line = line.split(None, 1)
                    command = line[0]
                    try:
                        params = dict([i.split('=') for i in line[1].strip().split(' ; ')])
                    except IndexError:
                        params = {}
                    class_ = [i for i in CMDS for j in CMDS[i] if j == command][0]
                    prefix = self._EF.data_controller.datasets[dataset] if class_ == 'preprocessor' else\
                        self._EF.process_controller.process if class_ == 'process' else\
                        self._EF.__getattribute__(class_)
                except Exception as e:
                    print(repr(e))
                    raise SyntaxError(f"Wrong command '{command}', line {num + 1}")

                for arg_name, value in params.items():
                    try:
                        annotation = prefix.__getattribute__(command).__annotations__[arg_name]
                        if type(annotation) in (tuple, list, set):
                            value = value.strip('[({})]').split(',')
                            ln = len(annotation)
                            if ln == 1:
                                params[arg_name] = [cast_value(annotation[0], i.strip()) for i in value]
                            else:
                                params[arg_name] = [cast_value(annotation[i], value[i.strip()]) for i in range(ln)]
                        else:
                            params[arg_name] = cast_value(annotation, value)
                    except Exception as e:
                        if value == 'None':
                            params[arg_name] = None
                        else:
                            print(repr(e))
                            raise ValueError(f'Wrong type of parameters ({arg_name}={value}), line {num + 1}')

                try:
                    prefix.__getattribute__(command)(**params)
                except Exception as e:
                    print(repr(e))
                    raise Exception(f'Line {num + 1} raises an exception')

                if command == 'get_dataset':
                    dataset = params['name']



# d = ef.process_controller.process.get_target('validation')
# d = d.flatten()
# f = ef.process_controller.process.get_forecasts('arima_000', 'validation', alpha=0.05)[0]
# if isinstance(forecast, dict):
#     conf_int = forecast['conf_int']
#     steps = forecast['steps']
#     start = forecast['start']
#     forecast = forecast['forecast']
# i = ef.process_controller.process._get_intervals_from_residuals(ef.process_controller.process.get_residuals('arima_000'), f, 0.05)
# l_, h_ = np.array([*zip(*i)][0]), np.array([*zip(*i)][1])
# print((sum(d > h_) + sum(d < l_)) / len(d))
