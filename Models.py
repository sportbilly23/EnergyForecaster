import dill
import os
import sys
import time
import tempfile
import warnings
from scipy.stats import norm

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

    flatten_models = ['linear', 'gru', 'lstm', 'rnn']
    recursive_models = ['gru', 'lstm', 'rnn']
    change_size_components = ['conv1d', 'convT1d', 'adaptiveavg1d', 'adaptivemax1d', 'avg1d', 'max1d', 'lp1d',
                              'circular1d', 'constant1d', 'reflection1d', 'replication1d', 'zero1d']

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
            'mish': {
                'class': nn.Mish,
                'params': {
                }
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
            },
            'tanh': {
                'class': nn.Tanh,
                'params': {
                }
            }
        },
        'model': {
            'conv1d': {
                'class': nn.Conv1d,
                'params': {
                    'in_channels': {
                        'in_channels': True,
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'out_channels': {
                        'out_channels': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'kernel_size': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 3
                    },
                    'stride': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'padding': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 0
                    },
                    'padding_mode': {
                        'required': True,
                        'class': str,
                        'class_container': ('zeros', 'reflect', 'replicate', 'circular'),
                        'value': 'zeros'
                    },
                    'dilation': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 1
                    },
                    'groups': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    }
                }
            },
            'gru': {
                'class': nn.GRU,
                'params': {
                    'input_size': {
                        'input_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'hidden_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'num_layers': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'batch_first': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'dropout': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'bidirectional': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'convT1d': {
                'class': nn.ConvTranspose1d,
                'params': {
                    'in_channels': {
                        'in_channels': True,
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'out_channels': {
                        'out_channels': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'kernel_size': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 3
                    },
                    'stride': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'padding': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 0
                    },
                    'output_padding': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 0
                    },
                    'padding_mode': {
                        'required': True,
                        'class': str,
                        'class_container': ('zeros', 'reflect', 'replicate', 'circular'),
                        'value': 'zeros'
                    },
                    'dilation': {
                        'required': True,
                        'class': (int, tuple),
                        'class_container': int,
                        'value': 1
                    },
                    'groups': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    }
                }
            },
            'lstm': {
                'class': nn.LSTM,
                'params': {
                    'input_size': {
                        'input_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'hidden_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'num_layers': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'batch_first': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'dropout': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'bidirectional': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'proj_size': {
                        'required': True,
                        'class': int,
                        'value': 0
                    }
                }
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
                        'value': True
                    }
                }
            },
            'rnn': {
                'class': nn.RNN,
                'params': {
                    'input_size': {
                        'input_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'hidden_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'num_layers': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'nonlinearity': {
                        'required': True,
                        'class': str,
                        'class_container': ('tanh', 'relu'),
                        'value': 'tanh'
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'batch_first': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'dropout': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'bidirectional': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'transformer': {
                'class': nn.Transformer,
                'params': {
                    'd_model': {
                        'input_size': True,
                        'required': True,
                        'class': int,
                        'value': 512
                    },
                    'nhead': {
                        'required': True,
                        'class': int,
                        'value': 8
                    },
                    'num_encoder_layers': {
                        'required': True,
                        'class': int,
                        'value': 6
                    },
                    'num_decoder_layers': {
                        'required': True,
                        'class': int,
                        'value': 6
                    },
                    'dim_feedforward': {
                        'required': True,
                        'class': int,
                        'value': 2048
                    },
                    'dropout': {
                        'required': True,
                        'class': float,
                        'value': 0.1
                    },
                    'activation': {
                        'required': True,
                        'class': str,
                        'class_members': ['relu', 'gelu'],
                        'value': 'relu'
                    },
                    'custom_encoder': {
                        'required': True,
                        'class': (nn.TransformerDecoder, type(None)),
                        'value': None
                    },
                    'custom_decoder': {
                        'required': True,
                        'class': (nn.TransformerDecoder, type(None)),
                        'value': None
                    },
                    'layer_norm_eps': {
                        'required': True,
                        'class': float,
                        'value': 1e-5
                    },
                    'batch_first': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'norm_first': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    }
                }
            }
        },
        'pooling': {
            'adaptiveavg1d': {
                'class': nn.AdaptiveAvgPool1d,
                'params': {
                    'output_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 16
                    }
                }
            },
            'adaptivemax1d': {
                'class': nn.AdaptiveMaxPool1d,
                'params': {
                    'output_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 16
                    }
                }
            },
            'avg1d': {
                'class': nn.AvgPool1d,
                'params': {
                    'kernel_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 3
                    },
                    'stride': {
                        'required': True,
                        'class': (int, type(None)),
                        'value': None
                    },
                    'padding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    },
                    'ceil_mode': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'count_include_pad': {
                        'required': True,
                        'class': bool,
                        'value': True
                    }
                }
            },
            'max1d': {
                'class': nn.MaxPool1d,
                'params': {
                    'kernel_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 3
                    },
                    'stride': {
                        'required': True,
                        'class': (int, type(None)),
                        'value': None
                    },
                    'padding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    },
                    'dilation': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'ceil_mode': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'return_indices ': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'lp1d': {
                'class': nn.LPPool1d,
                'params': {
                    'norm_type': {
                        'required': True,
                        'class': int,
                        'value': 2
                    },
                    'kernel_size': {
                        'output_size': True,
                        'required': True,
                        'class': int,
                        'value': 3
                    },
                    'stride': {
                        'required': True,
                        'class': (int, type(None)),
                        'value': None
                    },
                    'ceil_mode': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            }
        },
        'padding': {
            'circular1d': {
                'class': nn.CircularPad1d,
                'params': {
                    'paddding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    }
                }
            },
            'constant1d': {
                'class': nn.ConstantPad1d,
                'params': {
                    'paddding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    }
                },
                'value': {
                    'required': True,
                    'class': float,
                    'value': 0.
                }
            },
            'reflection1d': {
                'class': nn.ReflectionPad1d,
                'params': {
                    'paddding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    }
                }
            },
            'replication1d': {
                'class': nn.ReplicationPad1d,
                'params': {
                    'paddding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    }
                }
            },
            'zero1d': {
                'class': nn.ZeroPad1d,
                'params': {
                    'paddding': {
                        'required': True,
                        'class': int,
                        'value': 0
                    }
                }
            }
        },
        'normalization': {
            'layernorm': {
                'class': nn.LayerNorm,
                'params': {
                    'normalized_shape': {
                        'input_size': True,
                        'required': True,
                        'class': (int, list),
                        'class_container': int,
                        'value': 100
                    },
                    'eps': {
                        'required': True,
                        'class': float,
                        'value': 1e-5
                    },
                    'elementwise_affine': {
                        'required': True,
                        'class': bool,
                        'value': True
                    },
                    'bias': {
                        'required': True,
                        'class': bool,
                        'value': True
                    }
                }
            },
            'localresponsenorm': {
                'class': nn.LocalResponseNorm,
                'params': {
                    'size ': {
                        'required': True,
                        'class': int,
                        'value': 1
                    },
                    'alpha': {
                        'required': True,
                        'class': float,
                        'value': 1e-4
                    },
                    'beta': {
                        'required': True,
                        'class': float,
                        'value': 0.75
                    },
                    'k': {
                        'required': True,
                        'class': float,
                        'value': 1.
                    }
                }
            }
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
            },
            'l1': {
                'class': nn.L1Loss,
                'params': {
                    'reduction': {
                        'required': True,
                        'class': str,
                        'class_members': ['none', 'mean', 'sum'],
                        'value': 'mean'
                    }
                }
            },
            'huber': {
                'class': nn.HuberLoss,
                'params': {
                    'reduction': {
                        'required': True,
                        'class': str,
                        'class_members': ['none', 'mean', 'sum'],
                        'value': 'mean'
                    },
                    'delta': {
                        'required': True,
                        'class': float,
                        'value': 1.
                    }
                }
            },
            'smoothl1': {
                'class': nn.SmoothL1Loss,
                'params': {
                    'reduction': {
                        'required': True,
                        'class': str,
                        'class_members': ['none', 'mean', 'sum'],
                        'value': 'mean'
                    },
                    'beta': {
                        'required': True,
                        'class': float,
                        'value': 1.
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
            },
            'adadelta': {
                'class': optim.Adadelta,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1.
                    },
                    'rho': {
                        'required': True,
                        'class': float,
                        'value': 0.9
                    },
                    'eps': {
                        'required': True,
                        'class': float,
                        'value': 1e-6
                    },
                    'weight_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
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
                    'differentiable': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'adagrad': {
                'class': optim.Adagrad,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1e-2
                    },
                    'lr_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'eps': {
                        'required': True,
                        'class': float,
                        'value': 1e-10
                    },
                    'weight_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'initial_accumulator_value': {
                        'required': True,
                        'class': float,
                        'value': 0.
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
                    'differentiable': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'adamw': {
                'class': optim.AdamW,
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
                        'value': 1e-2
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
            },
            'sparseadam': {
                'class': optim.SparseAdam,
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
                    'maximize': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'adamax': {
                'class': optim.Adamax,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 2e-3
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
                    }
                }
            },
            'nadam': {
                'class': optim.NAdam,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 2e-3
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
                    'momentum_decay': {
                        'required': True,
                        'class': float,
                        'value': 4e-3
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
                    },
                    'decoupled_weight_decay': {
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
                    }
                }
            },
            'radam': {
                'class': optim.RAdam,
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
                    'decoupled_weight_decay': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
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
                    }
                }
            },
            'sgd': {
                'class': optim.SGD,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1e-3
                    },
                    'momentum': {
                        'required': True,
                        'class': float,
                        'value': 0.,
                    },
                    'weight_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'dampening': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'nesterov': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'maximize': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
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
            },
            'asgd': {
                'class': optim.ASGD,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1e-2
                    },
                    'lambd': {
                        'required': True,
                        'class': float,
                        'value': 1e-4,
                    },
                    'alpha': {
                        'required': True,
                        'class': float,
                        'value': 0.75,
                    },
                    't0': {
                        'required': True,
                        'class': float,
                        'value': 1e-6,
                    },
                    'weight_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'maximize': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
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
                    }
                }
            },
            'lbfgs': {
                'class': optim.LBFGS,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1.
                    },
                    'max_iter': {
                        'required': True,
                        'class': int,
                        'value': 20,
                    },
                    'max_eval': {
                        'required': True,
                        'class': int,
                        'value': 25
                    },
                    'tolerance_grad': {
                        'required': True,
                        'class': float,
                        'value': 1e-7
                    },
                    'tolerance_change': {
                        'required': True,
                        'class': float,
                        'value': 1e-9
                    },
                    'history_size': {
                        'required': True,
                        'class': int,
                        'value': 100
                    },
                    'line_search_fn': {
                        'required': True,
                        'class': (str, type(None)),
                        'class_members': ['strong_wolfe', None],
                        'value': None
                    }
                }
            },
            'rprop': {
                'class': optim.Rprop,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1e-2
                    },
                    'etas': {
                        'required': True,
                        'class': tuple,
                        'value': (0.5, 1.2),
                        'class_container': (float, float)
                    },
                    'step_sizes': {
                        'required': True,
                        'class': tuple,
                        'value': (1e-6, 50.),
                        'class_container': (float, float)
                    },
                    'maximize': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
                    },
                    'differentiable': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'rmsprop': {
                'class': optim.RMSprop,
                'params': {
                    'lr': {
                        'required': True,
                        'class': float,
                        'value': 1e-2
                    },
                    'momentum': {
                        'required': True,
                        'class': float,
                        'value': 0.,
                    },
                    'weight_decay': {
                        'required': True,
                        'class': float,
                        'value': 0.
                    },
                    'alpha': {
                        'required': True,
                        'class': float,
                        'value': 0.99
                    },
                    'eps': {
                        'required': True,
                        'class': float,
                        'value': 1e-8
                    },
                    'maximize': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'foreach': {
                        'required': True,
                        'class': (bool, type(None)),
                        'value': None
                    },
                    'centered': {
                        'required': True,
                        'class': bool,
                        'value': False
                    },
                    'differentiable': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            }
        },
        'dropout': {
            'dropout': {
                'class': nn.Dropout,
                'params': {
                    'p': {
                        'required': True,
                        'class': float,
                        'value': 0.5
                    },
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            },
            'alphadropout': {
                'class': nn.AlphaDropout,
                'params': {
                    'p': {
                        'required': True,
                        'class': float,
                        'value': 0.5
                    },
                    'inplace': {
                        'required': True,
                        'class': bool,
                        'value': False
                    }
                }
            }
        }
    }

    @staticmethod
    def search(component):
        """
        Return component from TorchInterface.comps by name
        :param component: (str) name of the component
        :return: (dict) component's interface details
        """
        for category in TorchInterface.comps:
            if component in TorchInterface.comps[category]:
                return TorchInterface.comps[category][component]

    @staticmethod
    def params(component):
        """
        Return parameters of a torch component
        :param component:  (str) torch component
        :return: (str) string with component's parameters
        """
        return ', '.join(TorchInterface.search(component)['params'].keys())

    @staticmethod
    def list_models():
        """
        returns supported models
        :return: (str) string with all supported models
        """
        return ', '.join(TorchInterface.comps['model'].keys())

    @staticmethod
    def list_activation_funcs():
        """
        returns supported activation functions
        :return: (str) string with all supported activation functions
        """
        return ', '.join(TorchInterface.comps['activation_func'].keys())

    @staticmethod
    def list_dropouts():
        """
        returns supported dropout features
        :return: (str) string with all supported dropout features
        """
        return ', '.join(TorchInterface.comps['dropout'].keys())

    @staticmethod
    def list_paddings():
        """
        returns supported paddings
        :return: (str) string with all supported paddings
        """
        return ', '.join(TorchInterface.comps['padding'].keys())

    @staticmethod
    def list_normalizations():
        """
        returns supported normalizations
        :return: (str) string with all supported normalizations
        """
        return ', '.join(TorchInterface.comps['normalization'].keys())

    @staticmethod
    def list_loss_funcs():
        """
        returns supported loss functions
        :return: (str) string with all supported loss functions
        """
        return ', '.join(TorchInterface.comps['loss_func'].keys())

    @staticmethod
    def list_poolings():
        """
        returns supported poolings
        :return: (str) string with all supported poolings
        """
        return ', '.join(TorchInterface.comps['pooling'].keys())

    @staticmethod
    def list_optimizers():
        """
        returns supported optimizers
        :return: (str) string with all supported optimizers
        """
        return ', '.join(TorchInterface.comps['optimizer'].keys())


class SelectItem(nn.Module):
    """
    class to select certain item from a list
    used for rnn models to take only the output for the forward call
    """
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

    def __repr__(self):
        return f'Select({self.item_index})'


class VotingModel:
    def __init__(self, name, filenames):
        self.name = name
        self.model_filenames = filenames
        self._number_of_models = len(filenames)
        self.results = {'resid': self._get_residuals()}

    def _get_residuals(self):
        resids = []
        for i in range(self._number_of_models):
            with open(self.model_filenames[i], 'rb') as f:
                model = dill.load(f)
            resids.append(model.get_residuals(torch_best_valid=False).flatten())
        return np.mean(np.vstack(resids), axis=0)

    def get_validation_residuals(self, data, target):
        """
        Returns validation residuals of the model
        :param data: (numpy.ndarray) data for validation
        :param target: (numpy.ndarray) target data for validation
        :return: (numpy.ndarray) residuals of the model
        """
        resids = []
        for i in range(self._number_of_models):
            with open(self.model_filenames[i], 'rb') as f:
                model = dill.load(f)
            if not isinstance(model.results, dict) or 'valid_resid' not in model.results:
                if isinstance(data, type(None)) or isinstance(target, type(None)):
                    raise ValueError(
                        'no validation residuals founded, you need to run function with validation dataset and target')
                else:
                    model.results.update(
                        {'valid_resid':
                             target.flatten() - model.get_forecasts(data)['forecast'].flatten()
                         }
                    )

            resids.append(model.results['valid_resid'].flatten())
        self.results.update({'valid_resid': np.mean(np.vstack(resids), axis=0)})
        return self.results['valid_resid']

    def get_forecasts(self, data, start=0, steps=None, alpha=None, torch_best_valid=True,
                      torch_best_loss_if_no_valid=True, intervals_from_validation=True):
        """
        Returns forecasts for given data
        :param data: (numpy.ndarray) data for creating forecasts
        :param start: (int) starting point for the plot
        :param steps: (int) steps to depict on the plot
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :param intervals_from_validation: (bool) True to calculate intervals from validation data
        :return: (dict) forecasts and forecast details
        """
        forecasts = []
        conf_int = []
        if isinstance(steps, type(None)):
            steps = data.shape[0]

        for i in range(self._number_of_models):
            with open(self.model_filenames[i], 'rb') as f:
                model = dill.load(f)
            forecast = model.get_forecasts(data, start, steps, alpha, torch_best_valid=torch_best_valid,
                                          torch_best_loss_if_no_valid=torch_best_loss_if_no_valid,
                                          intervals_from_validation=intervals_from_validation)
            forecasts.append(forecast['forecast'].flatten())
            conf_int.append(forecast['conf_int'])
            start = max(start, forecast['start'])
            steps = min(steps, forecast['steps'])
            alpha = forecast['alpha']

        return {'forecast': np.mean(np.vstack(forecasts), axis=0), 'start': start, 'steps': steps,
                'alpha': alpha, 'conf_int': np.mean(conf_int, axis=0)}


class TorchModel:

    def __init__(self, input_size, device, components):
        self._ti = TorchInterface
        self.input_size = input_size
        self.channels = 1
        self.output_size = -1
        self.device = device
        self.model = nn.Sequential().to(device)
        self.optimizer = None
        self.loss_func = None
        self.validation_func = None

        self.n_epochs_fitted = 0
        self.loss_history = []
        self.validation_history = []
        self.epoch_times = []

        self.best_loss_epoch = None
        self.best_loss = np.inf
        self.best_loss_state = None

        self.best_validation_epoch = None
        self.best_validation = None
        self.best_validation_state = None

        self.add_components(components)
        self._flatten_parameters_of_rnns()

    def _flatten_parameters_of_rnns(self):
        """
        flatten parameters of rnn models, for memory compression and warnings avoidance
        :return: (None)
        """
        for model in self.model:
            if model.__class__ in [TorchInterface.comps['model'][m]['class'] for m in TorchInterface.recursive_models]:
                model.flatten_parameters()

    def _raise_parameter_value_error(self, class_, comp, param):
        """
        raise exception for parameter wrong values
        :param class_: (class) component's class
        :param comp:  (str) component's name
        :param param: (str) component's parameter
        :return: (None)
        """
        raise ValueError(f'value error for ({class_.__name__}) variable {comp} --{param}')

    def _check_type(self, cat, comp, param, value):
        """
        check validity of parameter value
        :param cat: (str) category name of the component
        :param comp: (str) name of the component
        :param param: (str) name of the parameter
        :param value: (var) value of the parameter
        :return: (bool) True if parameter value is valid
        """
        tag = self._ti.comps[cat][comp]['params'][param]
        if isinstance(value, tag['class']):
            if isinstance(value, (tuple, list, set)):
                for i in range(len(value)):
                    try:
                        if not isinstance(list(value)[i], tag['class_container'][i]):
                            self._raise_parameter_value_error(tag['class'], comp, param)
                    except IndexError:
                        raise ValueError(
                            f'too many values to unpack in {comp} --{param} (expected {len(tag["class_container"])})')
                    except TypeError:
                        if not isinstance(list(value)[i], tag['class_container']):
                            self._raise_parameter_value_error(tag['class'], comp, param)
            elif 'class_members' in tag and value not in tag['class_container']:
                self._raise_parameter_value_error(tag['class'], comp, param)
        else:
            self._raise_parameter_value_error(tag['class'], comp, param)
        return True

    def _is_input_size(self, cat, comp, param):
        """
        check if given parameter is used for component's input size
        :param cat: (str) name of component's category
        :param comp: (str) name of the component
        :param param: (str) name of the parameter
        :return: (bool) True if given parameter is used for component's input size
        """
        tag = self._ti.comps[cat][comp]['params'][param]
        return 'input_size' in tag and tag['input_size']

    def _is_output_size(self, cat, comp, param):
        """
        check if given parameter is used for component's output size
        :param cat: (str) name of component's category
        :param comp: (str) name of the component
        :param param: (str) name of the parameter
        :return: (bool) True if given parameter is used for component's output size
        """
        tag = self._ti.comps[cat][comp]['params'][param]
        return 'output_size' in tag and tag['output_size']

    def _is_input_channel(self, cat, comp, param):
        """
        check if given parameter is used for component's input channels
        :param cat: (str) name of component's category
        :param comp: (str) name of the component
        :param param: (str) name of the parameter
        :return: (bool) True if given parameter is used for component's input channels
        """
        tag = self._ti.comps[cat][comp]['params'][param]
        return 'in_channels' in tag and tag['in_channels']

    def _is_output_channel(self, cat, comp, param):
        """
        check if given parameter is used for component's output channels
        :param cat: (str) name of component's category
        :param comp: (str) name of the component
        :param param: (str) name of the parameter
        :return: (bool) True if given parameter is used for component's output channels
        """
        tag = self._ti.comps[cat][comp]['params'][param]
        return 'out_channels' in tag and tag['out_channels']

    def add_components(self, components):
        """
        add a sequence of components to the neural network
        :param components: (list) list of components in format ['comp1', {'param1': value1, 'param1': value1},
        'comp2', {'param': value}, 'comp3', {}]
        :return: (None)
        """
        # previous_cat = None
        previous_comp = None
        for component, params in zip(components[::2], components[1::2]):

            cur_channels = self.channels
            cur_output_size = self.output_size
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

            # Flatten features if models need that
            if self.channels > 1 and component in TorchInterface.flatten_models:
                self.model.append(nn.Flatten())
                self.model.append(nn.Unflatten(1, (1, cur_channels * cur_output_size)))
                self.channels = 1

            for param in params:
                if self._is_input_size(category, component, param):
                    params[param] = self.input_size if cur_output_size == -1 else cur_channels * cur_output_size \
                        if component in TorchInterface.flatten_models else cur_output_size
                if self._is_output_size(category, component, param):
                    self.output_size = params[param]
                if self._is_input_channel(category, component, param):
                    params[param] = cur_channels
                if self._is_output_channel(category, component, param):
                    self.channels = params[param]

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
                instance = self._get_instance(category, component, params).to(self.device)
                self.model.append(instance)
                if component in TorchInterface.change_size_components:
                    if self.output_size == -1:
                        self.output_size = self.input_size
                    with torch.no_grad():
                        self.output_size = instance(
                            torch.rand((1, cur_channels, self.output_size)).to(self.device)).shape[-1]

            # Select the exit layer from the recursive models
            if component in TorchInterface.recursive_models:
                self.model.append(SelectItem(0))

            # previous_cat = category
            previous_comp = component
        print()

    def _get_instance(self, cat, comp, params):
        """
        creates an instance of the component
        :param cat: (str) name of component's category
        :param comp: (str) name of the component
        :param params: (dict) dictionary of component's parameters
        :return: (component) class instance of the component
        """
        return self._ti.comps[cat][comp]['class'](**params)

    def _from_numpy(self, data):
        """
        create torch tensor from numpy ndarray
        :param data: (numpy.ndarray) input data
        :return: (torch.Tensor) tensor with input data
        """
        return torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(self.device)

    def _get_dataloader(self, data, target, batch_size, shuffle=False):
        """
        Returns a dataloader for torch models
        :param data: (numpy.ndarray) dataset for the model
        :param target: (numpy.ndarray) target data for the model
        :param batch_size: (int) batch size of dataloader
        :param shuffle: (bool) True to shuffle data rows
        :return: (DataLoader) dataloader for torch models
        """
        dt = TensorDataset(self._from_numpy(data), self._from_numpy(target))
        return DataLoader(dt, batch_size=batch_size, shuffle=shuffle)

    def train(self, data, target, n_epochs, batch_size, shuffle=False, print_progress=True, time_interval=0.1,
              validation_data=None, validation_target=None, func=None, minimizing_func=True):
        """
        Trains torch models
        :param data: (numpy.ndarray) dataset for the model
        :param target: (numpy.ndarray) target data for the model
        :param n_epochs: (int) number of epochs for training
        :param batch_size: (int) batch size for the constructio of the dataloader
        :param shuffle: (bool) True to random shuffle the order of data instances
        :param print_progress: (bool) True to print progress details to the console
        :param time_interval: (float) time interval for progress printing
        :param validation_data: (numpy.ndarray) dataset for the train validation
        :param validation_target: (numpy.ndarray) target dataset for the train validation
        :param func: (func) evaluation function for validation
        :param minimizing_func: (bool) True if evaluation function is minimizing function
        :return: (None)
        """

        def _get_line_():
            """
            creates the training-progress-line
            :return: (str) training-progress-line
            """
            line_ = [f'\rEpoch: {old_epochs + epoch:>{epoch_len}}/{old_epochs + n_epochs} ({i / ln:5.2%})',
                     f'loss: {last_mean:1.5f}']
            if len(self.loss_history) > 0:
                line_.append(f'best_loss: {self.best_loss:1.5f} (epoch {self.best_loss_epoch + 1})')
            line_.append(f'elapsed time: {time_}')
            if not isinstance(validation_data, type(None)):
                line_.append(f'validation ({func.__name__}): {f"{validation:1.5f}" if validation else "-"}, '
                             f'best validation: {f"{self.best_validation:1.5f}" if not isinstance(self.best_validation, type(None)) else "-"} '
                             f'(epoch {f"{self.best_validation_epoch + 1}" if not isinstance(self.best_validation_epoch, type(None)) else "-"})')
            return ', '.join(line_)

        self._flatten_parameters_of_rnns()
        best_func = np.argmin if minimizing_func else np.argmax
        old_epochs = self.n_epochs_fitted

        validation = None

        dataloader = self._get_dataloader(data, target, batch_size, shuffle)
        if not isinstance(validation_data, type(None)):
            if isinstance(self.validation_func, type(None)):
                self.validation_func = func
            else:
                if self.validation_func.__class__ != func.__class__:
                    func = self.validation_func
                    warnings.warn(f'validation function is already set as {self.validation_func.__class__}')
            val_data = self._from_numpy(validation_data)
            with torch.no_grad():
                validation = func(validation_target, self.model(val_data).squeeze(1).cpu().numpy())

        line = ''

        ln = len(dataloader)
        epoch_len = len(str(n_epochs))

        start_time = time.time()
        cycle_time = start_time

        for epoch in range(1, n_epochs + 1):
            epoch_losses = []
            for i, (x, y) in enumerate(dataloader, start=1):
                preds = self.model(x)
                loss = self.loss_func(preds, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                last_mean = np.mean(epoch_losses)

                if print_progress:
                    now = time.time()
                    if now - cycle_time >= time_interval:
                        try:
                            not_first = self.epoch_times[1:] if len(self.epoch_times) > 1 else self.epoch_times
                            time_ = utils.timedelta_to_str((n_epochs - epoch + 1 - i / ln)
                                                           * sum(not_first) / len(not_first))
                        except ZeroDivisionError:
                            time_ = 'unknown'
                        prev_line = len(line)
                        line = _get_line_()
                        print(line, end=f"{' ' * (len(line) - prev_line)}")
                        sys.stdin.flush()

            if not isinstance(validation_data, type(None)):
                with torch.no_grad():
                    validation = func(validation_target, self.model(val_data).squeeze(1).cpu().numpy())

            self.loss_history.append(np.mean(epoch_losses))
            if len(self.loss_history) == 0 or self.loss_history[-1] < self.best_loss:
                self.best_loss_epoch = epoch + old_epochs
                self.best_loss = self.loss_history[-1]
                self.best_loss_state = self.clone_weights()

            if not isinstance(validation_data, type(None)):
                self.validation_history.append(validation)
                self.best_validation_epoch = best_func(self.validation_history)
                if self.best_validation_epoch == old_epochs + epoch - 1:
                    self.best_validation = self.validation_history[-1]
                    self.best_validation_state = self.clone_weights()

            self.n_epochs_fitted += 1

            end_time = time.time()
            self.epoch_times.append(end_time - start_time)
            start_time = end_time

        if print_progress:
            prev_line = len(line)
            line = _get_line_()
            print(line, end=f"{' ' * (len(line) - prev_line)}")
            sys.stdin.flush()
            print()

    def clone_weights(self):
        """
        create a copy of model's state_dict (weights)
        :return: (dict) state_dict copy
        """
        dct = self.model.state_dict()
        return {lab: dct[lab].clone() for lab in dct}

    def predict(self, data):
        """
        calculates predictions for the given data
        :param data: (numpy.ndarray) data
        :return: (torch.Tensor) tensor with predictions
        """
        with torch.no_grad():
            self._flatten_parameters_of_rnns()
            return self.model(self._from_numpy(data))


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
    TorchModel = TorchModel

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
        elif isinstance(self.model, TransformerModel):
            return None
        elif isinstance(self.model, MLPRegressor):
            return None
        elif isinstance(self.model, TorchModel):
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
        elif isinstance(self.model, TorchModel):
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
        elif isinstance(self.model, TorchModel):
            return None

    def _get_intervals(self, resids, forecast, alpha):
        """
        Calculates confidense intervals from mean and standard deviation of the residuals
        :param resids: (numpy.ndarray) residuals of the model
        :param forecast: (numpy.ndarray) forecasts for the training data
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :return: (list) list of tuples with down and up limit of the prediction confidence
        """
        mn = np.mean(resids)
        std = np.std(resids)
        from_ = norm.ppf(alpha / 2, mn, std)
        to_ = norm.ppf(1 - alpha / 2, mn, std)
        conf_int = [(i + from_, i + to_) for i in forecast]
        return conf_int

    def get_forecasts(self, data, start=0, steps=None, alpha=None, intervals_from_validation=True,
                      torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns forecasts for given data
        :param data: (numpy.ndarray) data for creating forecasts
        :param start: (int) starting point at data
        :param steps: (int) number of steps to forecast
        :param alpha: (float) alpha for prediction intervals (0 < alpha <= .5)
        :param intervals_from_validation: (bool) True to calculate intervals from validation data
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
        :return: (dict) forecasts and forecast details
        """
        if not steps:
            steps = len(data)

        if alpha:
            resids = self.get_validation_residuals() if intervals_from_validation else \
                self.get_residuals(torch_best_valid=torch_best_valid,
                                   torch_best_loss_if_no_valid=torch_best_loss_if_no_valid)

        if isinstance(self.model, SARIMAX) and self.results:
            forecast_results = self.results.get_forecast(exog=data, steps=len(data))
            forecast = forecast_results.predicted_mean[start: start + steps]
            intervals = self._get_intervals(resids, forecast.flatten(), alpha) if alpha else []

            return {'forecast': forecast, 'start': start, 'steps': steps, 'alpha': alpha,
                    'conf_int': intervals}

        elif isinstance(self.model, RandomForestRegressor) and self.results:
            forecast_results = self.model.predict(data)
            forecast = forecast_results[start: start + steps]
            intervals = self._get_intervals(resids, forecast.flatten(), alpha) if alpha else []

            return {'forecast': forecast, 'start': start, 'steps': steps, 'alpha': alpha,
                    'conf_int': intervals}

        elif self.model == TransformerModel:
            model = self._open_darts_model()
            if isinstance(steps, type(None)):
                steps = model.output_chunk_length
            if start > model.output_chunk_length:
                start = 0
            if start + steps > model.output_chunk_length:
                steps = model.output_chunk_length - start
            preds = model.predict(model.output_chunk_length)
            forecast = preds[start: start + steps].all_values().flatten()

            intervals = self._get_intervals(resids, forecast.flatten(), alpha) if alpha else []

            return {'forecast': forecast, 'start': start, 'steps': steps, 'alpha': alpha,
                    'conf_int': intervals}

        elif isinstance(self.model, MLPRegressor):
            forecast = self.model.predict(data)[start: start + steps]
            intervals = self._get_intervals(resids, forecast.flatten(), alpha) if alpha else []

            return {'forecast': forecast, 'start': start, 'steps': steps, 'alpha': alpha,
                    'conf_int': intervals}

        elif isinstance(self.model, TorchModel):
            if torch_best_valid:
                try:
                    self.model.model.load_state_dict(self.model.best_validation_state)
                except TypeError as e:
                    if torch_best_loss_if_no_valid:
                        torch_best_valid = False
                    else:
                        raise e
            if not torch_best_valid:
                self.model.model.load_state_dict(self.model.best_loss_state)

            forecast = self.model.predict(data)[start: start + steps].squeeze(1).cpu().numpy()
            intervals = self._get_intervals(resids, forecast.flatten(), alpha) if alpha else []

            return {'forecast': forecast, 'start': start, 'steps': steps, 'alpha': alpha,
                    'conf_int': intervals}

        raise NameError('Model type is not defined')

    def get_residuals(self, torch_best_valid=True, torch_best_loss_if_no_valid=True):
        """
        Returns residuals of the model
        :param torch_best_valid: (bool) True to use the best validation epoch (for TorchModel only)
        :param torch_best_loss_if_no_valid: (bool) True to use the best loss epoch if no validation was calculated (for
                                                   TorchModel only)
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
        elif isinstance(self.model, TorchModel):
            # return self.results['resid']['best_valid'] if 'valid' in self.results['resid'] and torch_best_valid else \
            #     self.results['resid']['best_loss'] if torch_best_loss_if_no_valid else None
            return self.results['valid_resid'] if 'valid_resid' in self.results and torch_best_valid else \
                self.results['resid'] if torch_best_loss_if_no_valid else None

    def get_validation_residuals(self, data=None, target=None):
        """
        Returns validation residuals of the model
        :param data: (numpy.ndarray) data for validation
        :param target: (numpy.ndarray) target data for validation
        :return: (numpy.ndarray) residuals of the model
        """
        if not isinstance(self.results, dict) or 'valid_resid' not in self.results:
            if isinstance(data, type(None)) or isinstance(target, type(None)):
                raise ValueError(
                    'no validation residuals founded, you need to run function with validation dataset and target')
            else:
                self.results.update(
                    {'valid_resid':
                         target.flatten() - self.get_forecasts(data)['forecast'].flatten()
                     }
                )

        return self.results['valid_resid']

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

    def extend_fit(self, data, target=None, n_epochs=1, validation_data=None, validation_target=None):
        """
        Extends model training for models that
        :param data: (numpy.ndarray) training dataset
        :param target: (numpy.ndarray) target dataset
        :param n_epochs: number of epochs for training
        :param validation_data: (numpy.ndarray) dataset for validation (for Torchmodel only)
        :param validation_target: (numpy.ndarray) target dataset for validation (for Torchmodel only)
        :return: (None)
        """
        if self.results:
            if isinstance(self.model, MLPRegressor):
                target = target.flatten()
                for _ in range(n_epochs):
                    self.results['extra_fit'] += 1
                    self.model.partial_fit(data, target, **self.fit_params)
                self.results['resid'] = target - self.model.predict(data)

            elif isinstance(self.model, TorchModel):
                self.fit(data, target, n_epochs=n_epochs, validation_data=validation_data,
                         validation_target=validation_target)
            else:
                raise ValueError('extend_fit not defined')

        else:
            raise ValueError('model is not fitted')

    def fit(self, data, target=None, scale=None, n_epochs=1, validation_data=None, validation_target=None):
        """
        Trains the model
        :param data: (numpy.ndarray) training dataset
        :param target: (numpy.ndarray) target dataset
        :param scale: (numpy.ndarray) scale
        :param n_epochs: number of epochs for training
        :param validation_data: (numpy.ndarray) dataset for validation (for Torchmodel only)
        :param validation_target: (numpy.ndarray) target dataset for validation (for Torchmodel only)
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
            self.model = TransformerModel

        elif isinstance(self.model, MLPRegressor):
            target = target.flatten()
            self.model.fit(data, target, **self.fit_params)
            self.results = {'resid': target - self.model.predict(data), 'extra_fit': 0}

        elif isinstance(self.model, TorchModel):
            self.model.train(data, target, n_epochs, validation_data=validation_data,
                             validation_target=validation_target, **self.fit_params)
            state_dict = self.model.clone_weights()
            self.model.model.load_state_dict(self.model.best_loss_state)
            # self.results = {'resid': {'best_loss': target - self.get_forecasts(data)['forecast']}}
            self.results = {'resid': target - self.get_forecasts(data)['forecast']}
            if not isinstance(self.model.best_validation_state, type(None)):
                self.model.model.load_state_dict(self.model.best_validation_state)
                # self.results['resid'].update(
                #     {'best_valid': validation_target - self.get_forecasts(validation_data)['forecast']})
                self.results.update({'valid_resid':
                                         validation_target - self.get_forecasts(validation_data)['forecast']})
            self.model.model.load_state_dict(state_dict)

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
