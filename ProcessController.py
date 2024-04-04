import utils
from DictNoDupl import DictNoDupl
import numpy as np
import pytz
from scipy.stats import norm
from Models import *


CMDS = {'process_controller': ['set_process', 'get_process', 'update_process', 'insert_data'],
        'data_controller': ['import_csv', 'get_dataset', 'update_dataset'],
        'preprocessor': ['log', 'log2', 'log10', 'ln', 'exp', 'exp2', 'boxcox', 'limit_output', 'minmax', 'standard',
                         'robust', 'differentiate', 'croston_method', 'to_timestamp', 'weekend', 'weekday', 'monthday',
                         'day_hour', 'year_day', 'year_week', 'year_month', 'month_weekdays', 'public_holidays',
                         'lagged_series', 'fill_backward', 'fill_forward', 'fill_linear', 'downgrade_data_frequency',
                         'attach_scale', 'make_target']}


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
        self.attributes = DictNoDupl()
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

    def _get_evaluation(self, name, func, data_part):
        """
        Creates evaluation statistics for a model
        :param name: (str) The name of the model
        :param func: (func) Function from Statistics.py to use evaluation (MAPE, MAE, RMSE, MSE, R-squared)
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) Evaluations for the models
        """
        actual = self.get_target(data_part).flatten()
        forecast = self.get_forecasts(name, data_part)

        return func(actual, forecast)

    def mape(self, name, data_part='train'):
        """
        Returns Mean Absolute Percentage Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) MAPE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.mape, data_part)

    def wmape(self, name, data_part='train'):
        """
        Returns weighted Mean Absolute Percentage Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) wMAPE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.wmape, data_part)

    def mae(self, name, data_part='train'):
        """
        Returns Mean Absolute Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) MAE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.mae, data_part)

    def rmse(self, name, data_part='train'):
        """
        Returns Root Mean Square Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) RMSE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.rmse, data_part)

    def mse(self, name, data_part='train'):
        """
        Returns Mean Square Error evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) MSE evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.mse, data_part)

    def r2(self, name, data_part='train'):
        """
        Returns R-Squared evaluation for a model
        :param name: (str) The name of the model
        :param data_part: (str) The part of data to use for the statistics ('train', 'validation', 'test')
        :return: (dict) R-squared evaluation for a model
        """
        return self._get_evaluation(name, self._EF.results_statistics.r2, data_part)

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

    def box_pierce(self, name, lags=[10]):
        """
        Box-Pierce portmanteau test for a model
        :param name: (str) The name of the model
        :param lags: (int or list(int)) Lags to return test values
        :return: (tuple(float)) Box-Pierce q-value and p-value
        """
        return self._EF.results_statistics.box_pierce(self.get_residuals(name), lags)

    def ljung_box(self, name, lags=[10]):
        """
        Ljung-Box portmanteau test for a model
        :param name: (str) The name of the model
        :param lags: (int or list(int)) Lags to return test values
        :return: (tuple(float)) Ljung-Box q-value and p-value
        """
        return self._EF.results_statistics.ljung_box(self.get_residuals(name), lags)

    def get_forecasts(self, name, data_part='train', start=0, steps=None, alpha=None):
        """
        Returns forecasts for a model
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
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
        return model.get_forecasts(data, start, steps, alpha)

    def get_residuals(self, name):
        """
        Returns residuals of a model by name
        :param name: (str) name of the model
        :return: (numpy.ndarray) residuals of the model
        """
        model = self._EF.data_controller._get_model(name)
        return model.get_residuals()

    def get_all_residuals(self):
        """
        Returns residuals of all models
        :return: (dict) The residuals of all the models
        """
        resids = {}
        for name in self.models:
            resid = self.get_residuals(name)
            resids.update({name: resid if not isinstance(resid, type(None)) else 'not fitted'})
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
        scale_str = utils.timestamp_to_date_str(scale, self.timezone)
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

    def get_intervals_from_residuals(self, resids, forecast, alpha):
        """
        Calculates confidense intervals from mean and standard deviation of the residuals
        :param resids: (numpy.ndarray) residuals of the model
        :param forecast: (numpy.ndarray) forecasts for the training data
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :return: (list) list of tuples with down and up limit of the prediction confidence
        """
        mn = np.mean(resids)
        std = np.std(resids)
        from_ = norm.ppf(alpha, mn, std)
        to_ = norm.ppf(1 - alpha, mn, std)
        conf_int = [(i + from_, i + to_) for i in forecast]
        return conf_int

    def plot_forecast(self, name, data_part='train', start=0, steps=None, alpha=None, axes=None,
                      intervals_from_residuals=True):
        """
        Gets forecasts of a model and supplies results_visualizer to create a plot
        :param name: (str) name of the model
        :param data_part: (str) The part of target data to get('train', 'validation', 'test')
        :param start: (int) starting point for the plot
        :param steps: (int) steps to depict on the plot
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :param intervals_from_residuals: (bool) True to create prediction interval from residuals distribution
        :return: (pyplot.axes) axes of the plot
        """
        not_alpha = isinstance(alpha, type(None))
        if not not_alpha:
            alpha = min(alpha, 1 - alpha)
        if not_alpha or 0 < alpha <= .5:
            forecast = self.get_forecasts(name, data_part=data_part, start=start, steps=steps,
                                          alpha=None if intervals_from_residuals else alpha)
            conf_int = []
            if alpha:
                if intervals_from_residuals:
                    resids = self.get_residuals(name)
                    conf_int = self.get_intervals_from_residuals(resids, forecast, alpha)
                else:
                    forecast, conf_int = forecast

            actual = self.get_target(data_part)[start: start + len(forecast)]
            scale = utils.timestamp_to_date_str(self.get_scale(data_part)[start: start + len(forecast)], self.timezone)
            self._EF.results_visualizer.plot_forecast(scale, actual, forecast, conf_int,
                                                      name=f'{name}' + ('' if isinstance(alpha, type(None)) else
                                                      f' - confidence {1 - alpha:1.0%}'), axes=axes)
        else:
            raise ValueError('Alpha must be a float number between 0 and 1')


class ProcessController:
    """
    Manages processes for Energy Forecaster
    """

    def __init__(self, ef=None):
        self._EF = ef
        self.process = None

    def set_model(self, model, name):
        """
        Insert model to the current process
        :param model: A well-defined model
        :param name: (str) A name for the model
        :param interface: (str) The interface to control the model training and the
        :return:
        """
        self._EF.data_controller._set_model(name, model)
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
            if isinstance(model.results, type(None)):
                model.fit(self.process.get_data(), self.process.get_target())
                self._EF.data_controller._update_model(model)

    def set_process(self, name: str, lags: int = 0, black_lags: int = 0, target_length: int = 1,
                    update_file: bool = False, train: float = .6, validation: float = .2, test: float = .2):
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

    def insert_data(self, dataset: str, columns: [str], change_to_new_tzone: bool = True, no_lags: bool = True):
        """
        Addind data to the process
        :param no_lags: (bool) False to create data lags
        :param dataset: (str) Name of the dataset
        :param columns: (list(str)) Names of the columns to be added
        :param change_to_new_tzone: (bool) If True, it changes the timezone with respect of new data timezone
        :return: (None)
        """
        if self.process:
            self._EF.data_controller._import_data_to_process(dataset, columns, self.process,
                                                             change_to_new_tzone=change_to_new_tzone, no_lags=no_lags)

    def data_summary(self, columns=None):
        """
        Returns a summary for the data
        :param columns: (list(str)) list of the column names
        :return: (str) printable summary of data
        """
        return self._EF.data_controller.data_summary(self.process, columns)

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
# i = ef.process_controller.process.get_intervals_from_residuals(ef.process_controller.process.get_residuals('arima_000'), f, 0.05)
# l_, h_ = np.array([*zip(*i)][0]), np.array([*zip(*i)][1])
# print((sum(d > h_) + sum(d < l_)) / len(d))
