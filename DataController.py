import warnings
import os
import h5py
from DictNoDupl import DictNoDupl
import numpy as np
import dill
from DTable import DTable
from utils import arrays_are_equal


SIGNATURE = ['__SIGNATURE__', 'Energy Forecaster Framework']
PROCESS_DATASET = '__DATASETS__'
PROCESS_MODEL = '__MODELS__'
PROCESS_MODEL_RESULTS = '__RESULTS__'
PROCESSES = '__PROCESSES__'
DTYPES = {'object': h5py.special_dtype(vlen=bytes)}


class DataController:
    """
    Manages EnergyForecaster files
    """
    def __init__(self, ef, folderpath):
        """
        Creates the main folder which contains a file of references as well as the subfolders "Files", "Models" and
        "Results".
        """
        self._EF = ef
        self._INITIAL_ATTRIBUTES = {'scale': None, 'transformations': [], 'units': 'units', 'comments': '',
                                    'target': False, 'lag': 0}

        folderpath = os.path.abspath(folderpath)

        # Check for folder validity
        ef, processes = self._check_filename(folderpath)

        self.path = folderpath
        self.name = os.path.split(self.path)[-1]
        self.processes = DictNoDupl(processes)
        self._EF.process_controller.datasets = self._EF.preprocessor.datasets = self.datasets = DictNoDupl()
        self.__check_changes = False
        self.__new_file = False

        if not ef:
            os.mkdir(folderpath)
            os.mkdir(os.path.join(folderpath, 'data'))
            os.mkdir(os.path.join(folderpath, 'models'))
            with h5py.File(os.path.join(self.path, self.name + '.h5'), 'w') as f:
                f.attrs[SIGNATURE[0]] = SIGNATURE[1]

    def _check_filename(self, folderpath):
        """
        Check if path is valid or if there is an old valid instance in the folder
        :param folderpath: (str) Path of new or existing EF
        :return: Energy Forecaster or None
        """
        path, name = os.path.split(folderpath)
        ef = False
        processes = DictNoDupl()

        if not os.path.isdir(path):
            raise Exception(f"'{folderpath}' is not a valid path")
        else:
            if os.path.isdir(folderpath):

                h5 = os.path.join(path, name, name + '.h5')
                if not os.path.isfile(h5):
                    raise Exception(f"Not a valid folder ({folderpath}). The h5 file is missing.")
                else:
                    with h5py.File(h5, 'r') as f:
                        if f.attrs[SIGNATURE[0]] != SIGNATURE[1]:
                            raise Exception(f"'{h5}' file is not valid")
                        else:
                            ef = True
                            processes = DictNoDupl({p: self._get_process_by_name(p, f) for p in f.keys()})

                data = os.path.join(path, name, 'data')
                if not os.path.isdir(data):
                    raise Exception(f"Not a valid folder ({folderpath}). The 'data' folder is missing.")

                models = os.path.join(path, name, 'models')
                if not os.path.isdir(models):
                    raise Exception(f"Not a valid folder ({folderpath}). The 'models' folder is missing.")

        return ef, processes

    def _get_process_by_name(self, name, f):
        """
        Get process from file by name
        :param name: (str) Name of the process
        :param f: (h5py.File) The h5 file
        :return: (Process) The process from the file
        """
        data = DictNoDupl()
        if 'data' in f[name]:
            for key in f[name]['data'].keys():
                data.update({key: f[name]['data'][key][:]})
        target = DictNoDupl()
        if 'target' in f[name]:
            for key in f[name]['target'].keys():
                target.update({key: f[name]['target'][key][:]})
        scale = None
        if 'scale' in f[name]:
            scale = f[name]['scale'][:]
        target_length = self.get_attribute_object('target-length', f, name)
        train = self.get_attribute_object('train', f, name)
        validation = self.get_attribute_object('validation', f, name)
        test = self.get_attribute_object('test', f, name)
        models = self.get_attribute_object('models', f, name)
        timezone = self.get_attribute_object('timezone', f, name)
        lags = self.get_attribute_object('lags', f, name)
        black_lags = self.get_attribute_object('black-lags', f, name)
        return self._EF.process_controller._process_creation_from_file(name, target, data, scale, timezone, lags,
                                                                       black_lags, target_length, train, validation,
                                                                       test, models)

    @ staticmethod
    def _check_quotes(quote, split):
        """
        Checks split difference with quotes existence
        :param quote: (char) The character of the quotes
        :param split: (list[str]) The split of the line without quotes
        :return: list[str] The new split of the line (with quotes). Returns None if new split is not valid.
        """
        if not quote:
            return split
        quot_split = []
        current = ''
        for s in split:
            if current:
                if s.startswith(quote):
                    return None
                if s.endswith(quote):
                    current += s
                    quot_split.append(current[1: -1])
                    current = ''
            else:
                if s.startswith(quote) and s.endswith(quote):
                    quot_split.append(s[1: -1])
                elif s.endswith(quote):
                    return None
                elif s.startswith(quote):
                    current += s
                else:
                    quot_split.append(s)
        return None if current else quot_split

    def _check_dataset_name_availability(self, filename):
        """
        Check if a filename can be used
        :param filename: (str) The filename to check if already exists
        :return: (None) Raises an exception if a file exists with this name
        """
        datasets = self.get_dataset_names(full=True)
        if filename in datasets:
            raise FileExistsError(f"Destination file ({filename}) already exists. Please define another name.")

    def import_csv(self, filename: str, delimiter=',', quotes='', headline=True, encoding=None, skip=0, h5_name=None):
        """
        open a csv file with delimiter, quotechar and type automations
        :param filename: (str) Path/Filename of csv file to import
        :param delimiter: (str) Separator between members of a line (default ',')
        :param quotes: (str) Quotes that are used for members containing functioning characters (default '')
        :param headline: (bool) True if file contains headers (default True)
        :param encoding: (str) If file contains encoded text members (default None)
        :param skip: (int) Lines at the start of the file that contains no data (default 0)
        :param h5_name: (str) Name to use for storing the created h5 file or None to use the same (default None)
        :return: (None)
        """
        new_filename = f'{h5_name if h5_name else os.path.splitext(os.path.split(filename)[1])[0]}.h5'
        self._check_dataset_name_availability(filename)

        splits = self._get_splitted_string_lines(filename, quotes, delimiter, skip, encoding)

        table = self._create_npstructure(splits, headline)

        self._set_dataset(table, new_filename)

    def get_dataset_names(self, full=False):
        """
        Returns the names of all h5 files in data folder
        :param full: (bool) True value returns name and extension and False only name
        :return: (list[str]) Names of h5 files in data folder
        """
        return [f if full else os.path.splitext(f)[0]
                for f in os.listdir(os.path.join('e:\\test', 'data')) if f.endswith('.h5')]

    def is_dataset_changed(self, name):
        """
        Returns the changing status of a dataset
        :param name: (str) Name of the file (without extension)
        :return: (bool) True if dataset have been changed since last save
        """
        self.__check_changes = True
        ret = self.get_dataset(name)
        self.__check_changes = False
        return ret

    def _check_dataset_name(self, name):
        """
        Checks if filename exists. And returns full name
        :param name: (str) Name to be checked if it exists (with or without extension)
        :return: (str) Filename with extension
        """
        if name not in self.get_dataset_names() and self.__check_changes:
            raise KeyError('Key is not in current datasets')
        if name in self.get_dataset_names():
            name += '.h5'
        elif name in self.get_dataset_names(full=True):
            pass
        else:
            raise KeyError('No such dataset in the folder')
        return name

    def get_dataset(self, name, in_line=False):
        """
        Loads an h5 file to the memory
        :param name: (str) Name of the file to get in memory
        :return: (None) Put numpy.ndarray in self.datasets dictionary
        """
        filename = self._check_dataset_name(name)

        with h5py.File(os.path.join(self.path, 'data', filename), 'r') as f:
            columns = self.get_attribute_object('columns', f)
            dtypes = self.get_attribute_object('dtypes', f)
            attributes = DictNoDupl()
            for c in columns:
                attributes.update({c: {i: self.get_attribute_object(i, f, c)
                                       for i, j in f[c].attrs.items()}})

            table = np.zeros(len(f[columns[0]]), dtypes)
            for column in columns:
                table[column] = f[column][...]

            for col in columns:
                if dtypes[col] == np.object_:
                    for i, j in enumerate(table[col]):
                        table[col][i] = j.decode()

        if self.__check_changes:
            # Check column names
            if table.dtype.names != self.datasets[name].columns:
                return True
            # Check attributes
            for col in columns:
                attrs = self.datasets[name].attributes[col]
                for attr in attrs:
                    if attr == 'transformations':
                        if not arrays_are_equal(self._EF.preprocessor.reverse_trans(table[col],
                                                                                    attributes[col]['transformations']),
                                                self.datasets[name].reverse_trans(col)):
                            return True
                    else:
                        if attributes[col][attr] != attrs[attr]:
                            return True
            # Check arrays
            for col in table.dtype.names:
                return not arrays_are_equal(self.datasets[name][col], table[col])
        else:
            if not in_line:
                self.datasets.update({name: DTable(table, name, attributes, self._EF)})
            else:
                return DTable(table, name, attributes, self._EF)

    def set_dataset(self, name, new_name):
        """
        Saves a dataset from memory to a new file
        :param name: (str) The key of the dataset (name of the h5 file) in self.datasets dictionary
        :param new_name: (str) The name for the new h5 file to be created
        :return: (None)
        """
        _ = self._check_dataset_name(name)

        new_name = f'{os.path.splitext(os.path.split(new_name)[1])[0]}.h5'
        self._check_dataset_name_availability(new_name)

        self._set_dataset(self.datasets[name].data, new_name)

    def update_dataset(self, name):
        """
        Updates file with changes have been done in memory
        :param name: (str) The name of the file
        :return: (None)
        """
        filename = self._check_dataset_name(name)

        with h5py.File(os.path.join(self.path, 'data', filename), 'r+') as f:
            self._set_attribute_object(self.datasets[name].columns, 'columns', f)
            self._set_attribute_object(self.datasets[name].dtypes, 'dtypes', f)

            for column in [k for k in f.keys() if k not in self.datasets[name].columns]:
                del f[column]

            for column in self.datasets[name].columns:
                if column in f.keys() and f[column][:].dtype != self.datasets[name][column].dtype:
                    del f[column]
                if column in f.keys():
                    f[column][:] = self.datasets[name][column]
                else:
                    try:
                        f.create_dataset(column, data=self.datasets[name][column],
                                         dtype=DTYPES[str(self.datasets[name].data.dtype[column])])
                    except KeyError:
                        f.create_dataset(column, data=self.datasets[name][column],
                                         dtype=self.datasets[name].data.dtype[column])

                for attr in [k for k in f[column].attrs.keys() if k not in self.datasets[name].attributes[column]]:
                    del f[column].attrs[attr]

                for attr in self.datasets[name].attributes[column]:
                    self._set_attribute_object(self.datasets[name].attributes[column][attr], attr, f, column)

    def close_dataset(self, name):
        """
        Removes a dataset from memory
        :param name: (str) The key of the dataset in self.datasets dictionary
        :return: (None)
        """
        _ = self._check_dataset_name(name)

        if self.is_dataset_changed(name):
            yn = input("Dataset has been changed. Are you sure you want to close it? (Type 'yes' to close): ")
            if yn.upper() == 'YES':
                self.datasets.pop(name)
        else:
            self.datasets.pop(name)

    def _set_dataset(self, table, filename):
        """
        Creates the h5 file and stores the dataset and some initial attributes
        :param table: (numpy.ndarray) The dataset
        :param filename: (str) The name of the file to be created
        :return: (None)
        """
        with h5py.File(os.path.join(self.path, 'data', filename), 'w') as f:
            self._set_attribute_object(table.dtype.names, 'columns', f)
            self._set_attribute_object(table.dtype, 'dtypes', f)

            for column in table.dtype.names:
                try:
                    f.create_dataset(column, data=table[column], dtype=DTYPES[str(table.dtype[column])])
                except KeyError:
                    f.create_dataset(column, data=table[column], dtype=table.dtype[column])
                for attr in self._INITIAL_ATTRIBUTES:
                    self._set_attribute_object(self._INITIAL_ATTRIBUTES[attr], attr, f, column)

    def _set_attribute_object(self, obj, attribute, file, path='/'):
        """
        Stores an object (ie. list, class instance etc) to an attribute in an h5 file
        :param obj: (obj) Python object to be stored
        :param attribute: (str) Name of the attribute
        :param file: (h5py.File) File where the attribute will be stored
        :param path: (str) Path in the file
        :return: (None)
        """
        bts = dill.dumps(obj)
        file[path].attrs[attribute] = np.array([b.to_bytes() for b in bts])

    def get_attribute_object(self, attribute, f, path='/'):
        """
        Get an object from an attribute in an h5 file
        :param attribute: (str) Name of the attribute
        :param f: (h5py.File) File where the attribute is stored
        :param path: (str) Path in the file
        :return: (obj) Python object that is stored in a attribute
        """
        return dill.loads(b''.join([b'\x00' if i == b'' else i for i in f[path].attrs[attribute]]))

    # def set_scale

    def _create_npstructure(self, splits: list, headline: bool) -> np.ndarray:
        """
        Creates the numpy.ndarray object that contains all information of a csv file. If no headers concluded, it
        creates some 'Unnamed' ones. It manages also header name conflicts. definds the dtype for every column.
        :param splits: (list[str]) A list of strings that contains all information of a csv file
        :param headline: (bool) True if splits list contains headers in the first
        :return: (numpy.ndarray) csv file information in numpy.ndarray formation
        """
        head = splits.pop(0) if headline else [f'unnamed-{1 + c}' for c in range(len(splits[0]))]
        head = self._fix_head_conflicts(head)

        series = np.array(splits)
        dtypes = self._get_types(series)
        table = self._create_table(series, head, dtypes)
        return table

    def _create_table(self, series, columns, dtypes, fix_strings=False):
        """
        Combines all information given from _create_npstructure to create the numpy.ndarray from a csv file
        :param series: (list[str]) A list of strings that contains all information of a csv file
        :param columns: (list[str]) List with the names of the columns
        :param dtypes: (list[numpy.dtype]) List with the dtypes of the columns
        :param fix_strings: (bool) False for create a h5 file from csv file. True for loading an h5 file from data
                                   folder (it needs conversion from bytes to strings)
        :return: (numpy.ndarray) csv file information in numpy.ndarray formation
        """
        table = np.zeros(shape=series.shape[:-1], dtype=list(zip(columns, dtypes)))
        for i, d in enumerate(dtypes):
            if fix_strings and d == np.object_:
                series[:, i] = [i.decode() for i in series[:, i]]
            table[columns[i]] = self._cast_string_with_nan(series[:, i], d)

        return table

    def _get_splitted_string_lines(self, filename: str, qt: str, dlm: str, sk: int, encoding: str) -> list:
        """
        Open a csv file and split lines and data with the given parameters
        :param filename: (str) Path/filename of the csv file
        :param qt: (str) Quote character
        :param dlm: (str) Delimiter character
        :param sk: (int) Number of skipped lines
        :param encoding: (str) Encoding of the file
        :return: (list[str]) List of strings with all the file information
        """
        newline = '\n'
        with open(filename, 'r', newline=newline, encoding=encoding) as f:
            if f.readline()[-2] == '\r':
                newline = '\r\n'

        with open(filename, 'r', newline=newline, encoding=encoding) as f:
            lines = f.readlines()

        splits = []
        for line in lines[sk:]:
            splits.append(self._check_quotes(qt, line[:-len(newline)].split(dlm)))

        return splits

    def _fix_head_conflicts(self, head):
        """
        Check about header conflicts and fix them automatically
        :param head: (list[str]) List of column names
        :return: (list[str]) List of fixed column names
        """
        un = 0
        for i, h in enumerate(head):
            h = head[i] = str(h)
            if not h:
                un += 1
                head[i] = f'unnamed-{un}'
            elif h in head[:i]:
                ind = 1
                while f'{h}-{ind}' in head[:i]:
                    ind += 1
                head[i] = f'{h}-{ind}'
        return head

    def _get_types(self, series):
        """
        Defines the dtypes of every column of the csv file
        :param series: (list[str]) List of strings that came out from a csv file
        :return: (list[numpy.dtype]) List of dtypes for the columns of the csv file
        """
        return [self._get_type(s) for s in series.T]

    def _cast_string_with_nan(self, series, dtype):
        """
        Converts a series of strings in to the appropriate type with respect to missing values
        :param series: (list[str]) List of strings represents to be converted
        :param dtype: (numpy.dtype) The appropriate type for the conversion
        :return: (numpy.ndarray) Numpy array with the appropriate dtype
        """
        return np.array([np.nan if i == '' else i for i in series], dtype=dtype)

    def _get_type(self, series):
        """
        Select the appropriate dtype for converting a list of strings
        :param series: (list[str]) List of strings
        :return: (numpy.dtype) The appropriate dtype for the conversion of the given list of strings
        """
        types = [(np.uint8, np.uint16, np.uint32, np.uint64),
                 (np.int16, np.int32, np.int64),
                 (np.float64,)]
        check = [True, True, True]

        all_ = ''.join(series)
        min_ = min(series)
        max_ = max(series)

        numeric = not all_.strip('0123456789.e-+')
        if not numeric:
            return np.dtypes.ObjectDType

        if '.' in all_ or '' in series:
            check[0] = check[1] = False
        elif '-' in all_:
            check[0] = False

        mask = np.random.choice(series, size=min(1000, series.shape[0]), replace=False)
        for c in range(3):
            if check[c]:
                try:
                    for tp in types[c]:
                        if c < 2:
                            info = np.iinfo(tp)
                            if tp(min_) < info.min or tp(max_) > info.max:
                                continue
                        self._cast_string_with_nan(mask, tp)
                        return tp
                except ValueError:
                    continue

        return np.dtypes.ObjectDType

    def _set_model(self, name, model, interface):
        """
        Creates a new file to store details of a model
        :param name: (str) The name of the model
        :param model: A well defined model
        :param interface: (str) A string to define the interface to use for the manipulation of the model
        :return: (None)
        """
        if name not in self.get_model_names():
            with open(os.path.join(self.path, 'models', name + '.pkl'), 'wb') as f:
                dill.dump({'name': name, 'model': model, 'interface': interface}, f)
        else:
            raise FileExistsError('Model name already exists')

    def _update_model(self, model):
        """
        Updates a model's file
        :param model: (dict) A dictionary with model's details
        :return: (None)
        """
        name = model['name']
        if name in self.get_model_names():
            with open(os.path.join(self.path, 'models', name + '.pkl'), 'wb') as f:
                dill.dump(model, f)
        else:
            raise FileExistsError('Model name already exists')

    def _get_model(self, name):
        """
        Get a model from the file
        :param name: (str) Name of the process to get
        :return: (dict) Dictionary with model's details
        """
        with open(os.path.join(self.path, 'models', name + '.pkl'), 'rb') as f:
            content = dill.load(f)
        return content

    def get_model_names(self, full=False):
        """
        Returns the names of all pkl files in model folder
        :param full: (bool) True value returns name and extension and False only name
        :return: (list[str]) Names of h5 files in data folder
        """
        return [f if full else os.path.splitext(f)[0]
                for f in os.listdir(os.path.join('e:\\test', 'models')) if f.endswith('.pkl')]

    def get_process_names(self):
        """
        Returns all names of the processes
        :return: (list(str)) List of names
        """
        return list(self.processes.keys())

    def get_process(self, name):
        """
        Get process with given name
        :param name: (str) Name of the process
        :return: (Process) The process
        """
        return self.processes[name]

    def set_process(self, process, update_file=False):
        """
        Store process to memory
        :param process: (Process) The process instance
        :param update_file: (bool) True to update the main file with the new process
        :return: (None)
        """
        self.processes.update({process.name: process})
        if update_file:
            self.update_process(process)

    def update_process(self, process):
        """
        Stores an updated process to the file
        :param process: (Process) The process instance
        :return: (None)
        """
        name = process.name
        with h5py.File(os.path.join(self.path, self.name + '.h5'), 'a') as f:
            if name not in f:
                f.create_group(f'{name}')

            if 'data' not in f[name]:
                f[name].create_group('data')
            if 'target' not in f[name]:
                f[name].create_group('target')

            for key in process.data.keys():
                try:
                    f[name]['data'].create_dataset(key, data=process.data[key], dtype=process.data[key].dtype)
                except ValueError:
                    f[name]['data'][key][:] = process.data[key]

            for key in process.target.keys():
                try:
                    f[name]['target'].create_dataset(key, data=process.target[key], dtype=process.target[key].dtype)
                except ValueError:
                    f[name]['target'][key][:] = process.target[key]

            try:
                f[name].create_dataset('scale', data=process.scale, dtype=process.scale.dtype)
            except ValueError:
                f[name]['scale'][:] = process.scale
            except AttributeError:
                pass

            self._set_attribute_object(process.target_length, 'target-length', f, name)
            self._set_attribute_object(process.train, 'train', f, name)
            self._set_attribute_object(process.validation, 'validation', f, name)
            self._set_attribute_object(process.test, 'test', f, name)
            self._set_attribute_object(process.models, 'models', f, name)
            self._set_attribute_object(process.timezone, 'timezone', f, name)
            self._set_attribute_object(process.lags, 'lags', f, name)
            self._set_attribute_object(process.black_lags, 'black-lags', f, name)

    def is_process_changed(self, process):
        """
        Checks if current process has been changed
        :param process: (Process) The process to be checked
        :return: (bool) True if current process have changes between RAM and file.
        """
        name = process['name']
        with h5py.File(os.path.join(self.path, self.name + '.h5'), 'r') as f:
            proc = self._get_process_by_name(name, f)
        proc.update({'name': name})
        return proc != process

    def remove_process(self, name):
        """
        Removes a process from the file
        :param name: (str) The name of the process
        :return: (None)
        """
        with h5py.File(os.path.join(self.path, self.name + '.h5'), 'r+') as f:
            del f[name]
        self.processes.pop(name)

    def _import_data_to_process(self, dataset, columns, process, change_to_new_tzone=True, no_lags=True):
        """
        Imports data in a process. Automatic creation of lagged data with respect in process settings
        :param dataset: (str) Name of the dataset
        :param columns: (list(str)) List of columns
        :param process: (Process) The process where data will be added
        :param change_to_new_tzone: (bool) If True, it changes the timezone with respect of new data timezone
        :return: (None)
        """
        if dataset in self.datasets and self.is_dataset_changed(dataset):
            warnings.warn("Dataset has temporary changes that will not be taken into account. You must update the file.")
        dataset = self.get_dataset(dataset, in_line=True)

        for col in columns:
            names = dataset[col].dtype.names
            col_scale = dataset.attributes[col]['scale']

            if not col_scale:
                raise ValueError(f"Column '{col}' has not a defined scale.")

            if change_to_new_tzone:
                process.timezone = dataset.attributes[col_scale]['timezone']

            if isinstance(process.scale, type(None)):
                process.scale = dataset[col_scale]
            else:
                if not arrays_are_equal(process.scale, dataset[col_scale]):
                    scale_format = np.in1d(process.scale, dataset[col_scale])
                    lst = list(process.data.keys())
                    for d in lst:
                        temp = process.data.pop(d)
                        process.data[d] = temp[scale_format]
                    process.scale = process.scale[scale_format]

            if dataset.attributes[col]['target']:
                if names:
                    for name in names:
                        process.target.update({f'{col}::{name}': dataset[col][name]})
                else:
                    process.target.update({f'{col}': dataset[col]})
            else:

                lag_cols = dataset.lagged_series(col, col, process.lags) if not no_lags else \
                    dataset.lagged_series(col, col, [0])
                for lag, lag_data in zip(process.lags if not no_lags else [0], lag_cols.dtype.names):
                    if names:
                        for name in names:
                            column = f'{col}_lag-{lag}::{name}'
                            process.data.update({column: lag_cols[lag_data]})  # dataset[col][name][cut - lag: -lag]})
                            # process.attributes.update({column: dataset.attributes[col].copy()})
                            # process.attributes[column]['lag'] = lag

                    else:
                        column = f'{col}_lag-{lag}'
                        process.data.update({column: lag_cols[lag_data]})  # dataset[col][cut - lag: -lag]})
                        # process.attributes.update({column: dataset.attributes[col].copy()})
                        # process.attributes[column]['lag'] = lag

