from statsmodels.tsa.statespace.sarimax import SARIMAX
from DictNoDupl import DictNoDupl
import numpy as np
import pytz


class Process:
    def __init__(self, name, target=DictNoDupl(), data=DictNoDupl(), scale=None, timezone=pytz.utc,
                 lags=1, black_lags=0, target_length=1, train=.6, validation=.2, test=.2, models=[]):
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

    def get_data(self, mode='train'):
        data = self._prepare_data(self.data)
        from_ = 0 if mode == 'train' \
            else data.shape[0] * self.train if mode == 'validation' \
            else data.shape[0] * (self.train + self.validation)
        to_ = data.shape[0] * self.train if mode == 'train' \
            else data.shape[0] * (self.train + self.validation) if mode == 'validation' \
            else data.shape[0]
        return data[int(from_):int(to_)]

    def get_target(self, mode='train'):
        data = self._prepare_data(self.target)
        from_ = 0 if mode == 'train' \
            else data.shape[0] * self.train if mode == 'validation' \
            else data.shape[0] * (self.train + self.validation)
        to_ = data.shape[0] * self.train if mode == 'train' \
            else data.shape[0] * (self.train + self.validation) if mode == 'validation' \
            else data.shape[0]
        return data[int(from_):int(to_)]

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
                       target_length, train, validation, test, models)

    def fit_models(self):
        """
        Training all models
        :return:
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
                                                     validation=validation / amount, test=test / amount),
                                             update_file=update_file)

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
            if yn.upper() == 'YES':
                self.process = None

    def update_process(self):
        """
        Stores the updated current process to the file
        :return:
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
        yn = input("Process will de deleted permanently. Are you sure you want to delete it? (Type 'yes' to close): ")
        if yn.upper() == 'YES':
            if name == self.process['name']:
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

