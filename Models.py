import os
import sys
import time
import tempfile
import types
import warnings

from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from darts.models import TransformerModel
from darts.timeseries import TimeSeries
from xarray import DataArray
from pandas import DatetimeIndex
import numpy as np
import utils


class TorchInterface:

    comps = {
        'activation_func': {
            'celu': {
                'class': nn.CELU,
                'params': {
                    'alpha ': {
                        'required': True,
                        'class': float,
                        'value': 1.
                    },
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'elu': {
                'class': nn.ELU,
                'params': {
                    'alpha': {
                        'required': True,
                        'class': float,
                        'value': 1.
                    },
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'gelu': {
                'class': nn.GELU,
                'params': {
                    'approximate': {
                        'required': True,
                        'class': str,
                        'class_members': ['none', 'tanh'],
                        'value': 'none'
                    }
                }
            },
            'glu': {
                'class': nn.GLU,
                'params': {
                    'dim': {
                        'required': True,
                        'class': int,
                        'value': -1
                    }
                }
            },
            'mish': {
                'class': nn.Mish
            },
            'prelu': {
                'class': nn.PReLU,
                'params': {
                    'num_parameters ': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'init': {
                        'required': True,
                        'class': float,
                        'value': 0.25
                    }
                }
            },
            'relu': {
                'class': nn.ReLU,
                'params': {
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'relu6': {
                'class': nn.ReLU6,
                'params': {
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'rrelu': {
                'class': nn.RReLU,
                'params': {
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'lower': {
                        'required': True,
                        'class': float,
                        'value': 1 / 8
                    },
                    'upper': {
                        'required': True,
                        'class': float,
                        'value': 1 / 3
                    }
                },
            },
            'selu': {
                'class': nn.SELU,
                'params': {
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'softmax': {
                'class': nn.Softmax,
                'params': {
                    'dim': {
                        'required': False,
                        'class': int,
                        'value': None
                    }
                }
            }
        },
        'model': {
            'conv1d': {
                'class': nn.Conv1d
            },
            'conv2d': {
                'class': nn.Conv2d
            },
            'gru': {
                'class': nn.GRU
            },
            'convT1d': {
                'class': nn.ConvTranspose1d
            },
            'convT2d': {
                'class': nn.ConvTranspose2d
            },
            'lstm': {
                'class': nn.LSTM
            },
            'Lconv1d': {
                'class': nn.LazyConv1d
            },
            'Lconv2d': {
                'class': nn.LazyConv2d
            },
            'LconvT1d': {
                'class': nn.LazyConvTranspose1d
            },
            'lconvT2d': {
                'class': nn.LazyConvTranspose2d
            },
            'Llinear': {
                'class': nn.LazyLinear
            },
            'linear': {
                'class': nn.Linear,
                'params': {
                    'in_features': {
                        'input_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'out_features': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'rnn': {
                'class': nn.RNN
            },
            'transformer': {
                'class': nn.Transformer
            }
        },
        'pooling': {

        },
        'padding': {

        },
        'normalization': {

        },
        'loss_func': {
            'mse': {
                'class': nn.MSELoss,
                'params': {
                    'reduction': {
                        'required': True,
                        'class': str,
                        'class_members': ['none', 'mean', 'sum'],
                        'value': 'mean'
                    }
                }
            }
        },
        'optimizer': {
            'adam': {
                'class': optim.Adam,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1e-3
                    },
                    'betas': {
                        'required': True,
                        'class': tuple,
                        'value': (0.9, 0.999),
                        'class_container': (float, float)
                    },
                    'eps': {
                        'required': True,
                        'class': float,
                        'value': 1e-8
                    },
                    'weight_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'amsgrad': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
                    },
                    'maximize': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'capturable': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'differentiable': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'fused': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
                    }

                }
            }
        },
        'dropout': {

        }
    }

    @staticmethod
    def list_models():
        return ', '.join(TorchInterface.comps['model'].keys())

    @staticmethod
    def list_activation_funcs():
        return ', '.join(TorchInterface.comps['activation_func'].keys())

    @staticmethod
    def list_dropouts():
        return ', '.join(TorchInterface.comps['dropout'].keys())

    @staticmethod
    def list_paddings():
        return ', '.join(TorchInterface.comps['padding'].keys())

    @staticmethod
    def list_normalizations():
        return ', '.join(TorchInterface.comps['normalization'].keys())

    @staticmethod
    def list_loss_funcs():
        return ', '.join(TorchInterface.comps['loss_func'].keys())

    @staticmethod
    def list_poolings():
        return ', '.join(TorchInterface.comps['pooling'].keys())

    @staticmethod
    def list_optimizers():
        return ', '.join(TorchInterface.comps['optimizer'].keys())


class TorchModel:

    def __init__(self, input_size):
        self._ti = TorchInterface
        self.input_size = input_size
        self.output_size = -1
        self.model = nn.Sequential()
        self.n_epochs_fitted = 0
        self.optimizer = None
        self.loss_func = None
        self.results = None

    def __raise_parameter_value_error(self, class_, comp, param):
        raise ValueError(f'value error for ({class_.__name__}) variable {comp} --{param}')

    def _check_type(self, cat, comp, param, value):
        tag = self._ti.comps[cat][comp]['params'][param]
        if isinstance(value, tag['class']):
            if isinstance(value, (tuple, list, set)):
                for i in range(len(value)):
                    try:
                        if not isinstance(list(value)[i], tag['class_container'][i]):
                            self.__raise_parameter_value_error(tag['class'], comp, param)
                    except IndexError:
                        raise ValueError(
                            f'too many values to unpack in {comp} --{param} (expected {len(tag["class_container"])})')
                    except TypeError:
                        if not isinstance(list(value)[i], tag['class_container']):
                            self.__raise_parameter_value_error(tag['class'], comp, param)
        else:
            self.__raise_parameter_value_error(tag['class'], comp, param)
        return True

    def _is_input_size(self, cat, comp, param):
        tag = self._ti.comps[cat][comp]['params'][param]
        return 'input_size' in tag and tag['input_size']

    def _is_output_size(self, cat, comp, param):
        tag = self._ti.comps[cat][comp]['params'][param]
        return 'output_size' in tag and tag['output_size']

    def add_components(self, components):
        previous_cat = None
        previous_comp = None
        for component, params in zip(components[::2], components[1::2]):

            for category in self._ti.comps:
                if component in self._ti.comps[category]:
                    break
            else:
                raise NameError(f'{component} is not defined')

            params = {p: v for p, v in params.items() if self._is_input_size(category, component, p) or
                      self._check_type(category, component, p, v)}

            params = {**{p: self._ti.comps[category][component]['params'][p]['value']
                         for p in self._ti.comps[category][component]['params']},
                      **params}

            for param in params:
                if self._is_input_size(category, component, param):
                    params[param] = self.input_size if self.output_size == -1 else self.output_size
                if self._is_output_size(category, component, param):
                    self.output_size = params[param]

            if category == 'optimizer' and self.n_epochs_fitted == 0:
                params = {**params, **{'params': self.model.parameters()}}
                self.optimizer = self._get_instance(category, component, params)
            elif category == 'loss_func' and self.n_epochs_fitted == 0:
                self.loss_func = self._get_instance(category, component, params)
            elif previous_comp in ['softmax']:
                warnings.warn(f'{previous_comp} usually is the last layer')
            # elif previous_cat in ['']
            # self._ti.comps[cat][comp]['class'](self._ti.comps[])
            elif category in ['model', 'activation_func']:
                self.model.append(self._get_instance(category, component, params))
        print()

    def _get_instance(self, cat, comp, params):
        return self._ti.comps[cat][comp]['class'](**params)

    def set_dataloader(self, data, target, batch_size, device='cpu', shuffle=False):
        """
        Returns a dataloader for torch models
        :param data: (numpy.ndarray) dataset for the model
        :param target: (numpy.ndarray) target data for the model
        :param batch_size: (int) batch size of dataloader
        :param device: (str) batch size of dataloader
        :param shuffle: (bool) True to shuffle data rows
        :return: (DataLoader) dataloader for torch models
        """
        dt = TensorDataset(torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(device),
                           torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device))
        return DataLoader(dt, batch_size=batch_size, shuffle=shuffle)

    def train(self, dataloader, n_epochs, print_progress=True, time_interval=0.1, validation_data=None, func=None):
        """
        Trains torch models
        :param dataloader: (torch.DataLoader) dataset/target-data iterator
        :param n_epochs: (int) number of epochs for training
        :return: (None)
        """
        loss_history = []
        best_loss = torch.inf
        best_epoch = 0
        epoch_times = []
        line = ''
        validation = None

        ln = len(dataloader)
        epoch_len = len(str(n_epochs))

        start_time = time.time()
        cycle_time = start_time

        for epoch in range(1, n_epochs + 1):
            for i, (x, y) in enumerate(dataloader, start=1):
                preds = self.model(x)
                loss = self.loss_func(preds, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_history.append(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch
                prev_line = len(line)

                if print_progress:
                    now = time.time()
                    if now - cycle_time >= time_interval:
                        try:
                            time_ = utils.timedelta_to_str((n_epochs - epoch + 1 - i / ln) * sum(epoch_times) /
                                                           len(epoch_times))
                        except ZeroDivisionError:
                            time_ = 'unknown'
                        line = (
                            f'\rEpoch: {epoch:>{epoch_len}}/{n_epochs} ({i / ln:5.2%}), '
                            f'loss: {loss}, best: {best_loss} (epoch {best_epoch}), elapsed time: {time_}, '
                            f'validation ({func.__name__}): {validation:1.4f}')
                        print(line, end=f"{' ' * (len(line) - prev_line)}")
                        sys.stdin.flush()

            if not isinstance(validation_data, type(None)):
                with torch.no_grad():
                    validation = func(validation_data['target'].cpu().numpy(),
                                      self.model(validation_data['data']).cpu().numpy())
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            start_time = end_time

        self.results = {'epoch_times': epoch_times, 'loss_history': loss_history,
                        'best_epoch': best_epoch, 'best_loss': best_loss}

    def show_architecture(self):
        pass

    def set_optimizer(self):
        pass

    def set_loss_func(self):
        pass


class Model:
    RandomForestRegressor = RandomForestRegressor
    SARIMAX = SARIMAX
    TransformerModel = TransformerModel
    torch = torch
    nn = nn
    optim = optim
    TensorDataset = TensorDataset
    DataLoader = DataLoader
    MLPRegressor = MLPRegressor

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
        elif isinstance(self.model, RandomForestRegressor):
            return None
        if isinstance(self.model, TransformerModel):
            return None
        elif isinstance(self.model, MLPRegressor):
            return None

    def aicc(self):
        """
        Calculates Akaike's Information Criterion corrected of residuals
        :return: (float) AICc score
        """
        if isinstance(self.model, SARIMAX):
            return self.model.aicc
        elif isinstance(self.model, RandomForestRegressor):
            return None
        if isinstance(self.model, TransformerModel):
            return None
        elif isinstance(self.model, MLPRegressor):
            return None

    def bic(self):
        """
        Calculates Bayesian Information Criterion of residuals
        :return: (float) BIC score
        """
        if isinstance(self.model, SARIMAX):
            return self.model.bic
        elif isinstance(self.model, RandomForestRegressor):
            return None
        if isinstance(self.model, TransformerModel):
            return None
        elif isinstance(self.model, MLPRegressor):
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
        if not steps:
            steps = len(data)
        if isinstance(self.model, SARIMAX) and self.results:
            forecast_results = self.results.get_forecast(exog=data, steps=len(data))
            forecast = forecast_results.predicted_mean[start: start + steps]
            if alpha:
                return {'forecast': forecast, 'start': start, 'steps': steps, 'alpha': alpha,
                        'conf_int': forecast_results.conf_int(alpha=alpha)[start: start + steps]}
            return forecast
        elif isinstance(self.model, RandomForestRegressor) and self.results:
            forecast_results = self.model.predict(data)
            forecast = forecast_results[start: start + steps]
            # TODO: confidence intervals
            # if alpha:
            #     err = np.sqrt(np.abs(fci.random_forest_error(self.model, train_data, data)))
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

            # TODO: confidence intervals
            return {'forecast': preds[start: start + steps].all_values().flatten(),
                    'start': start, 'steps': steps, 'alpha': None, 'conf_int': []}
        elif isinstance(self.model, MLPRegressor):
            forecast = self.model.predict(data)[start: start + steps]
            # TODO: confidence intervals
            return forecast

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
        elif isinstance(self.model, MLPRegressor):
            return self.results['resid']

    def _darts_timeseries(self, data, scale):
        """
        Returns data as darts.Timeseries
        :param data: (numpy.ndarray) input data
        :param scale: scale of the data
        :return: (darts.timeseries.TimeSeries) data as TimeSeries
        """
        scale = DatetimeIndex(scale * 1000000000)
        series = DataArray(np.expand_dims(data, 2), dims=['scale', 'component', 'sample'],
                           coords=dict(scale=scale),
                           attrs=dict(static_covariates=None, hierarchy=None))
        return TimeSeries(series)

    def extend_fit(self, data, target=None, n_epochs=1):
        if self.results:
            if isinstance(self.model, MLPRegressor):
                target = target.flatten()
                for _ in range(n_epochs):
                    self.results['extra_fit'] += 1
                    self.model.partial_fit(data, target, **self.fit_params)
                self.results['resid'] = target - self.model.predict(data)
            else:
                raise ValueError('extend_fit not defined')
        else:
            raise ValueError('model is not fitted')

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
            series = self._darts_timeseries(target, scale)

            past_cov = self._darts_timeseries(data, scale)

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
            self.model.predict
            self.model = TransformerModel
        elif isinstance(self.model, MLPRegressor):
            target = target.flatten()
            self.model.fit(data, target, **self.fit_params)
            self.results = {'resid': target - self.model.predict(data), 'extra_fit': 0}

    def _open_darts_model(self):
        """
        Opens a darts model
        :return: (darts model)
        """
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
        """
        Generates a random unique name for a temporary file and a second with 'ckpt' extension
        :return: (str) random unique name
        """
        while True:
            temp_file = tempfile.mktemp()
            if not os.path.isfile(f'{temp_file}.ckpt'):
                break
        return temp_file


    # class LR(nn.Module):
    #     def __init__(self, inputSize, layer1, outputSize):
    #         super(LR, self).__init__()
    #         self.linear1 = nn.Linear(inputSize, layer1)
    #         self.linear2 = nn.Linear(layer1, outputSize)
    #
    #     def forward(self, x):
    #         out = self.linear1(x)
    #         out = self.linear2(out)
    #         return out
    #
    # dl = _get_dataloader(ef.process_controller.process.get_data(), ef.process_controller.process.get_target(), 8, 'cuda', False)
    #
    # m = LR(54, 100, 1).to('cuda')
    # t = _train_torch_model(EFDataLoader(ef.process_controller.process.get_data(), ef.process_controller.process.get_target(), 'cuda'), m, optim.Adam(m.parameters()), nn.MSELoss(), 10)
    # with torch.no_grad():
    #     print(m(torch.tensor(X, dtype=torch.float32)))