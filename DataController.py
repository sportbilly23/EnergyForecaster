import os
import h5py
from DictNoDupl import DictNoDupl
import numpy as np
import pickle
from DTable import DTable

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

        folderpath = os.path.abspath(folderpath)

        # Check for folder validity
        ef, processes = self._check_filename(folderpath)

        self.path = folderpath
        self.name = os.path.split(self.path)[-1]
        self._EF.process_controller.processes = processes
        self._EF.process_controller.current_process = None
        self.current_datasets = DictNoDupl()
        self.__check_changes = False
        self.__new_file = False

        if not ef:
            os.mkdir(folderpath)
            os.mkdir(os.path.join(folderpath, 'data'))
            os.mkdir(os.path.join(folderpath, 'models'))
            # self._EF.visualizer.datasets = self._EF.statistics.datasets = self._EF.process_controller.datasets = \
            #     self._EF.preprocessor.datasets = self.datasets = DictNoDupl()
            with h5py.File(os.path.join(self.path, self.name + '.h5'), 'w') as f:
                f.attrs[SIGNATURE[0]] = SIGNATURE[1]

    @staticmethod
    def _check_filename(folderpath):
        """
        Check if path is valid or if there is an old valid instance in the folder
        :param folderpath: (str) Path of new or existing EF
        :return: Energy Forecaster or None
        """
        path, name = os.path.split(folderpath)
        ef = False
        processes = []

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
                            processes = list(f.keys())

                data = os.path.join(path, name, 'data')
                if not os.path.isdir(data):
                    raise Exception(f"Not a valid folder ({folderpath}). The 'data' folder is missing.")

                models = os.path.join(path, name, 'models')
                if not os.path.isdir(models):
                    raise Exception(f"Not a valid folder ({folderpath}). The 'models' folder is missing.")

        return ef, processes

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

    def dataset_is_changed(self, name):
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

    def get_dataset(self, name):
        """
        Loads an h5 file to the memory
        :param name: (str) Name of the file to get in memory
        :return: (None) Put numpy.ndarray in self.current_datasets dictionary
        """
        filename = self._check_dataset_name(name)

        with h5py.File(os.path.join(self.path, 'data', filename), 'r') as f:
            columns = self.get_attribute_object('columns', f)
            dtypes = self.get_attribute_object('dtypes', f)
            table = self._create_table(np.hstack([f[col][...].reshape(-1, 1) for col in columns]),
                                       columns, dtypes, fix_strings=True)

        if self.__check_changes:
            if table.dtype.names != self.current_datasets[name].columns:
                return True
            for col in table.dtype.names:
                try:
                    np.testing.assert_equal(self.current_datasets[name][col], table[col])
                except AssertionError:
                    return True
            return False
        else:
            self.current_datasets.update({name: DTable(table)})

    def set_dataset(self, name, new_name):
        """
        Saves a dataset from memory to a new file
        :param name: (str) The key of the dataset (name of the h5 file) in self.current_datasets dictionary
        :param new_name: (str) The name for the new h5 file to be created
        :return: (None)
        """
        _ = self._check_dataset_name(name)

        new_name = f'{os.path.splitext(os.path.split(new_name)[1])[0]}.h5'
        self._check_dataset_name_availability(new_name)

        self._set_dataset(self.current_datasets[name].data, new_name)

    def update_dataset(self, name):
        """

        :param name:
        :return:
        """
        filename = self._check_dataset_name(name)

        with h5py.File(os.path.join(self.path, 'data', filename), 'r+') as f:
            #TODO: Να ελέγξω για αλλαγές και αν υπάρχουν να κάνω τις αλλαγές στο αρχείο αφού πάρω επιβεβαίωση
            pass

    def close_dataset(self, name):
        """
        Removes a dataset from memory
        :param name: (str) The key of the dataset in self.current_datasets dictionary
        :return: (None)
        """
        _ = self._check_dataset_name(name)

        if self.dataset_is_changed(name):
            yn = input("Dataset has been changed. Are you sure you want to close it? (Type 'yes' to close): ")
            if yn.upper() == 'YES':
                self.current_datasets.pop(name)

    def _set_dataset(self, table, filename):
        """
        Creates the h5 file and stores the dataset and some initial attributes
        :param table: (numpy.ndarray) The dataset
        :param filename: (str) The name of the file to be created
        :return: (None)
        """
        with h5py.File(os.path.join(self.path, 'data', filename), 'w') as f:
            self._set_attribute_object(table.dtype.names, 'columns', f)
            dtypes = [tp[0] for tp in table.dtype.fields.values()]
            self._set_attribute_object(dtypes, 'dtypes', f)

            for column in table.dtype.names:
                try:
                    f.create_dataset(column, data=table[column], dtype=DTYPES[str(table.dtype[column])])
                except KeyError:
                    f.create_dataset(column, data=table[column], dtype=table.dtype[column])
                self._set_attribute_object(DictNoDupl(), 'scales', f, column)
                self._set_attribute_object(DictNoDupl(), 'transformations', f, column)
                self._set_attribute_object(DictNoDupl(), 'units', f, column)
                self._set_attribute_object(np.array([], dtype='S1'), 'comments', f, column)

    def _set_attribute_object(self, obj, attribute, file, path='/'):
        """
        Stores an object (ie. list, class instance etc) to an attribute in an h5 file
        :param obj: (obj) Python object to be stored
        :param attribute: (str) Name of the attribute
        :param file: (h5py.File) File where the attribute will be stored
        :param path: (str) Path in the file
        :return: (None)
        """
        bts = pickle.dumps(obj)
        file[path].attrs[attribute] = np.array([b.to_bytes() for b in bts])

    def get_attribute_object(self, attribute, f, path='/'):
        """
        Get an object from an attribute in an h5 file
        :param attribute: (str) Name of the attribute
        :param f: (h5py.File) File where the attribute will be stored
        :param path: (str) Path in the file
        :return: (obj) Python object that is stored in a attribute
        """
        return pickle.loads(b''.join([b'\x00' if i == b'' else i for i in f[path].attrs[attribute]]))

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
        Check about
        :param head:
        :return:
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

        :param series:
        :param accuracy:
        :return:
        """
        return [self._get_type(s) for s in series.T]

    def _cast_string_with_nan(self, ndarray, dtype):
        """

        :param ndarray:
        :param dtype:
        :return:
        """
        return np.array([np.nan if i == '' else i for i in ndarray], dtype=dtype)

    def _get_type(self, series):
        """

        :param series:
        :return:
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
