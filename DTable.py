import matplotlib.pyplot as plt
import numpy as np
import utils


class DTable:
    """
    Table that holds the data of an EF file and outputs graphs and statistics.
    """
    def __init__(self, data, name, attributes, _EF):
        self._EF = _EF
        self._INITIAL_ATTRIBUTES = _EF.data_controller._INITIAL_ATTRIBUTES
        self.name = name
        self.data = None
        self.columns = None
        self.dtypes = None
        self.attributes = attributes
        self._update_data(np.array(data))
        self._preprocessor = _EF.preprocessor
        self._visualizer = _EF.data_visualizer
        self._stats = _EF.data_statistics

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def log(self, column: str, base: float, assign: str = None, rename: str = None):
        """
        Data logarithmic transformation
        :param base: (int) Base of the logarithm transformation
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.log(base, self.data[column]),
                             assign, rename)

    def log2(self, column: str, assign: str = None, rename: str = None):
        """
        Data binary logarithmic transformation (base 2)
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.log2(self.data[column]),
                             assign, rename)

    def log10(self, column: str, assign: str = None, rename: str = None):
        """
        Data common logarithmic transformation (base 10)
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.log10(self.data[column]),
                             assign, rename)

    def ln(self, column: str, assign: str = None, rename: str = None):
        """
        Data natural logarithmic transformation (base e)
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.ln(self.data[column]),
                             assign, rename)

    def exp(self, column: str, assign: str = None, rename: str = None):
        """
        Data natural exponential transformation (base e)
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.exp(self.data[column]),
                             assign, rename)

    def exp2(self, column: str, assign: str = None, rename: str = None):
        """
        Data binary exponential transformation (base 2)
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.exp2(self.data[column]),
                             assign, rename)

    def boxcox(self, column: str, lamda: int = None, assign: str = None, rename: str = None):
        """
        Box-Cox transformation with lambda parameter
        :param column: (str) The label of the DTable's column to be transformed
        :param lamda: (float) lambda parameter of box-cox
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.boxcox(self.data[column], lamda),
                             assign, rename)

    def minmax(self, column: str, assign: str = None, rename: str = None):
        """
        Min-Max normalization
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.minmax(self.data[column]),
                             assign, rename)

    def standard(self, column: str, centered: bool = True, devarianced: bool = True, assign: str = None,
                 rename: str = None):
        """
        Standard normalization
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param centered: (bool) Subtracks mean value to center transformed data on axes
        :param devarianced: (bool) Divides by standard deviation to de-variance transformed data
        :param rename: (str) name to change
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.standard(self.data[column], centered=centered, devarianced=devarianced),
                             assign, rename)

    def robust(self, column: str, centered: bool = True, quantile_range: (float, float) = (.25, .75),
               assign: str = None, rename: str = None):
        """
        Robust normalization
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :param centered: (bool) Subtracks median value to center transformed data on axes
        :param quantile_range: (tuple(float)) Defines quantiles to de-variance transformed data
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return self._inplace(column,
                             self._preprocessor.robust(self.data[column], centered=centered,
                                                       quantile_range=quantile_range),
                             assign, rename)

    def differentiate(self, column: str, period: int = 1, assign: str = None, rename: str = None):
        """
        Differentiate data
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param period: (int) Period of the differentiation
        :param rename: (str) name to change
        :return: (numpy.ndarray) Transformed data
        """
        return self._inplace(column,
                             self._preprocessor.differentiate(self.data[column], period=period),
                             assign, rename)

    def croston_method(self, column: str, assign: str = None, rename: str = None):
        """
        Transform sparse series to inter-arrival/quantity series
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray) Transformed data
        """
        return self._inplace(column,
                             self._preprocessor.croston_method(self.data[column]),
                             assign, rename)

    def to_timestamp(self, column: str, form: str, tzone: str, assign: str = None, rename: str = None,
                     make_scale: bool = True):
        """
        Transforms strings of dates to timestamps
        :param column: (str) The label of the DTable's column to be transformed
        :param form: (str) Format of string data
        :param tzone: (str) Timezone label (eg. 'Europe/Athens')
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :param make_scale: (bool) True to define it as scale
        :return: (numpy.ndarray) Transformed data - Timestamps
        """
        if make_scale:
            self.attributes[column].update({'is_scale': True})
            self.attributes[column].update({'timezone': utils.get_tzinfo(tzone)})
        return self._inplace(column,
                             self._preprocessor.to_timestamp(self.data[column], form),
                             assign, rename)

    def weekend(self, column: str, assign: str = None, rename: str = None):
        """
        Weekday/weekend indicator
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray) Weekend indicators
        """
        return self._inplace(column,
                             self._preprocessor.weekend(self.data[column], self.attributes[column]['timezone']),
                             assign, rename)

    def weekday(self, column: str, mode: str = 'one-hot', assign: str = None, rename: str = None):
        """
        Weekday indicators/one-hot encoding/cyclical encoding
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :param rename: (str) name to change
        :return: (numpy.array) Weekday indicators
        """
        return self._inplace(column,
                             self._preprocessor.weekday(self.data[column],
                                                        self.attributes[column]['timezone'], mode=mode),
                             assign, rename)

    def monthday(self, column: str, mode: str = 'one-hot', assign: str = None, rename: str = None):
        """
        Monthday indicators/one-hot encoding/cyclical encoding
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :param rename: (str) name to change
        :return: (numpy.array) Monthday indicators
        """
        return self._inplace(column,
                             self._preprocessor.monthday(self.data[column],
                                                         self.attributes[column]['timezone'], mode=mode),
                             assign, rename)

    def day_hour(self, column: str, mode: str = 'one-hot', assign: str = None, rename: str = None):
        """
        Day hour indicators/one-hot encoding/cyclical encoding
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :param rename: (str) name to change
        :return: (numpy.array) Monthday indicators
        """
        return self._inplace(column,
                             self._preprocessor.day_hour(self.data[column],
                                                         self.attributes[column]['timezone'], mode=mode),
                             assign, rename)

    def year_day(self, column: str, mode: str = 'one-hot', assign: str = None, rename: str = None):
        """
        Day of year indicators/one-hot encoding/cyclical encoding
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :param rename: (str) name to change
        :return: (numpy.array) Day of year indicators
        """
        return self._inplace(column,
                             self._preprocessor.year_day(self.data[column],
                                                         self.attributes[column]['timezone'], mode=mode),
                             assign, rename)

    def year_week(self, column: str, mode: str = 'one-hot', assign: str = None, rename: str = None):
        """
        Week of year indicators/one-hot encoding/cyclical encoding
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :param rename: (str) name to change
        :return: (numpy.array) Week of year indicators
        """
        return self._inplace(column,
                             self._preprocessor.year_week(self.data[column],
                                                          self.attributes[column]['timezone'], mode=mode),
                             assign, rename)

    def year_month(self, column: str, mode: str = 'one-hot', assign: str = None, rename: str = None):
        """
        Month indicators/one-hot encoding/cyclical encoding
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :param rename: (str) name to change
        :return: (numpy.array) Month indicators
        """
        return self._inplace(column,
                             self._preprocessor.year_month(self.data[column],
                                                           self.attributes[column]['timezone'], mode=mode),
                             assign, rename)

    def month_weekdays(self, from_year_month: (int, int), to_year_month: (int, int), assign: str = None,
                       rename: str = None):
        """
        Creates monthly numbers of weekdays between two dates
        :param from_year_month: (tuple(int, int)) Starting year and month
        :param to_year_month:  (tuple(int, int)) Ending year and month
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray) Monthly numbers of weekdays between two dates
        """
        return self._inplace(None,
                             self._preprocessor.month_weekdays(from_year_month, to_year_month),
                             assign, rename)

    def public_holidays(self, column: str, holidays: [tuple], assign: str = None, rename: str = None):
        """
        Creates one-hot encoding for given list of holidays
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param holidays: (list(tuple)) List of tuples containing year, month and monthday of holidays
        :param rename: (str) name to change
        :return:(numpy.ndarray) One-hot encoding for national holidays
        """
        return self._inplace(column,
                             self._preprocessor.public_holidays(self.data[column], holidays,
                                                                self.attributes[column]['timezone']),
                             assign, rename)

    def lagged_series(self, column: str, name: str, lags: [int] = (1,), assign: str = None, rename: str = None):
        """
        Creates a number of lagged series for the input data
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param name: (str) To name the columns of lagged series
        :param lags: (tuple) Tuple with lags to be created
        :param rename: (str) name to change
        :return: (numpy.ndarray) Lagged series of the data
        """
        lag_data = self._inplace(column,
                                 self._preprocessor.lagged_series(self.data[column], name, lags=lags),
                                 assign, rename, True)
        if assign in ['add', 'inplace']:
            if assign == 'add':
                column = lag_data
            self.attributes[column]['lag'] = lags
            return None

        return lag_data

    def fill_backward(self, column: str, assign: str = None, rename: str = None):
        """
        Fills nan values with next value
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray) Data with filled the nan values (except last line's nans)
        """
        return self._inplace(column,
                             self._preprocessor.fill_backward(self.data[column]),
                             assign, rename)

    def fill_forward(self, column: str, assign: str = None, rename: str = None):
        """
        Fills nan values with previous value
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray)  (except first line's nans)
        """
        return self._inplace(column,
                             self._preprocessor.fill_forward(self.data[column]),
                             assign, rename)

    def fill_linear(self, column: str, assign: str = None, rename: str = None):
        """
        Fills nan values with linear interpolation between previous and next values
        :param column: (str) The label of the DTable's column to be transformed
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :param rename: (str) name to change
        :return: (numpy.ndarray)  (except first line's nans)
        """
        return self._inplace(column,
                             self._preprocessor.fill_linear(self.data[column]),
                             assign, rename)

    def _random_name(self, name, columns):
        """
        Creates name for added column so there are no conflicts with current dataset columns
        :param name: (str) Proposed name
        :param columns: list(str) Current column names
        :return: (str) Name for the new column
        """
        new_name = name
        while new_name in columns:
            new_name = f'{name}_{np.random.randint(int(1e8), int(1e9))}'
        return new_name

    def _inplace(self, column, data, assign, rename=None, return_new_name=False):
        """
        Controls column replacement and addition for new transformations
        :param column: (str) Label of the transformed column
        :param data: (numpy.ndarray) New, transformed column(s) to be added to the dataset
        :param rename: (str) name to change
        :param assign: (str) 'inplace'/'add'/None. Input column will be replaced if 'inplace' is selected. No impact
                             to the DTable if None is selected.
        :return: (numpy.ndarray, func) Returns the transformed data and (if possible) the reversed function
        """
        if isinstance(data, tuple):
            data, funcs = data
        else:
            funcs = None

        if assign in ['inplace', 'add']:

            new_column = self._random_name(column, self.columns) if assign == 'add' else column

            dtype = [(i, self.data.dtype[i]) for i in self.data.dtype.names
                     if not (assign == 'inplace' and i == column)]

            if rename:
                new_column = rename

            dtype.append((new_column, data.dtype))

            temp = np.zeros(shape=self.data.shape, dtype=dtype)
            for c in temp.dtype.names:
                if c == new_column:
                    temp[c] = data
                else:
                    temp[c] = self.data[c]

            if new_column not in self.attributes:
                if funcs:
                    self.attributes.update({new_column: self.attributes[column]})
                else:
                    self.attributes.update({new_column: self._INITIAL_ATTRIBUTES})

            if funcs:
                self.attributes[new_column]['transformations'].append(dict(zip(('func', 'rev'), funcs)))
            else:
                self.attributes[new_column]['transformations'] = []

            self._update_data(temp)
            if return_new_name and assign == 'add':
                return new_column
        else:
            return (data, funcs) if funcs else data

    def _update_data(self, data):
        """
        Updates data, columns and dtypes of them
        :param data: (numpy.ndarray) New data
        :return: (None)
        """
        self.data = data
        self.columns = self.data.dtype.names
        self.dtypes = data.dtype
        attributes_to_remove = [a for a in self.attributes if a not in self.columns]
        for attr in attributes_to_remove:
            self.attributes.pop(attr)

    def make_scale(self, column: str, tzone: str = None):
        """
        Defines column as scale
        :param column: (str) Label of th column to define as scale
        :param tzone: (str) Timezone label (eg. 'Europe/Athens')
        :return: (None)
        """
        self.attributes[column].update({'is_scale': True})
        self.attributes[column].update({'scale': None})
        if tzone:
            self.attributes[column].update({'timezone': utils.get_tzinfo(tzone)})

    def is_scale(self, column):
        """
        Checks if a column is scale
        :param column: (str) Label of th column to define as scale
        :return: (bool) True if a column is scale
        """
        return 'is_scale' in self.attributes[column] and self.attributes[column]['is_scale']

    def attach_scale(self, column: str, scale: str):
        """
        Attach a scale to a column
        :param column: (str) Label of the column to attach a scale
        :param scale: (str) Label of the scale column
        :return: (None)
        """
        if self.is_scale(scale) and not self.is_scale(column):
            self.attributes[column]['scale'] = scale
        else:
            raise ValueError('Method attach_scale needs a scale column to be attached to a non-scale column.')

    def make_target(self, column: str, disable: bool = False):
        """
        Defines column as target
        :param column: (str) Label of the column to define as target
        :param disable: (bool) True for removing column from target
        :return: (None)
        """
        self.attributes[column].update({'target': not disable})

    def detach_scale(self, column: str):
        """
        Detach scale from a column
        :param column: (str) Label of the column to detach scale
        :return: (None)
        """
        self.attributes[column]['scale'] = None

    def set_units(self, column: str, string: str):
        """
        Define the measurement units of a column
        :param column: (str) Label of the column
        :param string: (srt) Measurement units of a column
        :return: (None)
        """
        if not string:
            string = 'units'
        self.attributes[column]['units'] = str(string)

    def clear_comments(self, column: str):
        """
        Clears comments of a column
        :param column: (str) Label of the column to attach a scale
        :return: (None)
        """
        self.attributes[column]['comments'] = ''

    def add_comments(self, column: str, string: str):
        """
        Add a line to the comments of a column
        :param column: (str) Label of the column
        :param string: (str) Comment to be added to column's comments
        :return: (None)
        """
        comments = self.attributes[column]['comments']
        self.attributes[column]['comments'] = '\n'.join([comments, string]).strip('\n')

    def reverse_trans(self, column: str):
        """
        Returns data applying reverse transformation functions
        :param column: (str) Label of the column
        :return: (numpy.ndarray) Initial data of a column (before transformations)
        """
        return self._preprocessor.reverse_trans(self.data[column], self.attributes[column]['transformations'])

    def kpss(self, column: str, nlags: int = 'auto', regression: str = 'c'):
        """
        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) unit root test from statsmodels module
        :param column: (str) Label of the column
        :param nlags: (int) Number of lags to be used
        :param regression: (str) The null hypothesis for the KPSS test ('c', 'ct')
        :return: (list) KPSS test results from statsmodels module
        """
        return self._stats.kpss(self.data[column], nlags=nlags, regression=regression)

    def adf(self, column: str, maxlag: int = None, regression: str = 'c', autolag: str = 'AIC'):
        """
        Augmented Dickey-Fuller unit root test from statsmodels module
        :param column: (str) Label of the column
        :param maxlag: (int) Maximum lag which is included in test
        :param regression: (str) Constant and trend order to include in regression ('c', 'ct', 'ctt', 'n')
        :param autolag: (str) Method to use when automatically determining the lag length ('AIC', 'BIC', 't-stat', None)
        :return: (list) ADF test results from statsmodels module
        """
        return self._stats.adf(self.data[column], maxlag=maxlag, regression=regression, autolag=autolag)

    def pearson_correlation(self, column_a: str, column_b: str):
        """
        Returns Pearson correlation for the data of the two inserted columns
        :param column_a: (str) Label of the first column
        :param column_b: (str) Label of the second column
        :return: (float) Pearson correlation for the data of the two inserted columns
        """
        return self._stats.pearson_correlation(self.data[column_a], self.data[column_b])

    def spearman_correlation(self, column_a: str, column_b: str):
        """
        Returns Spearman correlation for the data of the two inserted columns
        :param column_a: (str) Label of the first column
        :param column_b: (str) Label of the second column
        :return: (float) Spearman correlation for the data of the two inserted columns
        """
        return self._stats.spearman_correlation(self.data[column_a], self.data[column_b])

    def zscore(self, column: str):
        """
        Calculates Z-score for the data of a column
        :param column: (str) Label of the column
        :return: (numpy.ndarray) Z-score for the data of the inserted column
        """
        return self._stats.zscore(self.data[column])

    def complexity_estimate(self, column: str):
        """
        Calculates Complexity Estimation
        :param column: (str) Label of the column
        :return: (float) Complexity Estimation
        """
        return self._stats.complexity_estimate(self.data[column])

    def mean_absolute_change(self, column: str):
        """
        Calculates Mean Absolute Change
        :param column: (str) Label of the column
        :return: (float) Mean Absolute Change
        """
        return self._stats.mean_absolute_change(self.data[column])

    def approximate_entropy(self, column: str, window: int, level: float):
        """
        Calculates Approximate Entropy
        :param column: (str) Label of the column
        :param window: (int) Length of a run of data
        :param level: (float) Filtering level
        :return: (float) Approximate Entropy
        """
        return self._stats.approximate_entropy(self.data[column], window=window, level=level)

    def mean(self, column: str, reverse_trans: bool = True):
        """
        Returns mean value of a column
        :param column: (str) Label of the column
        :param reverse_trans: (bool) Calculate on reverse transformed data
        :return: (float) Mean of the data
        """
        return self._stats.mean(self.reverse_trans(column) if reverse_trans else self.data[column])

    def std(self, column: str, dof: int = 1, reverse_trans: bool = True):
        """
        Returns the standard deviation of a column
        :param column: (str) Label of the column
        :param dof: (int) Degrees of freedom to use for the calculations
        :param reverse_trans: (bool) Calculate on reverse transformed data
        :return: (float) Standard deviation of the data
        """
        return self._stats.std(self.reverse_trans(column) if reverse_trans else self.data[column], dof=dof)

    def get_units(self, column: str):
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

    def scatter(self, column_1: str, column_2: str, reverse_transform: bool = True, axes: plt.axes = False):
        """
        Creates scatter-plot to visualize correlation between columns
        :param column_1: (str) Label of one column
        :param column_2: (str) Label of a second column
        :param reverse_transform: (bool) Plot with reverse transformed data
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        if self._compatible_data([column_1, column_2]) and self._no_scale_in_data([column_1, column_2]):
            if reverse_transform:
                return self._visualizer.scatter(self.reverse_trans(column_1), self.reverse_trans(column_2),
                                                (column_1, column_2),
                                                (self.attributes[column_1]['units'],
                                                 self.attributes[column_2]['units']),
                                                axes=axes)
            else:
                return self._visualizer.scatter(self.data[column_1], self.data[column_2],
                                                (column_1, column_2),
                                                (self.get_units(column_1), self.get_units(column_2)),
                                                axes=axes)

    def downgrade_data_frequency(self, column, freq, from_date=None, to_date=None, func=np.sum, reverse_transform=True):
        """
        Downgrade the frequency of a column data by applying a given function
        :param column: (str) Label of the column
        :param freq: (str) New data frequency ('year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'microsecond')
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param func: (func) Function to apply on data
        :param reverse_transform: (bool) Use reverse transformed data
        :return: (numpy.ndarray) Data with the new frequency
        """
        column_scale = self._get_scale(column)
        timezone = self.attributes[column_scale]['timezone']
        data = self.reverse_trans(column) if reverse_transform else self.data[column]
        return self._inplace(column,
                             self._preprocessor.downgrade_data_frequency(data, self.data[column_scale], freq,
                                                                         from_date, to_date, timezone, func),
                             assign=None)

    def _plot_calculate_data_scale_units(self, column, from_date, to_date, reverse_transform, freq, func):
        """
        Return data from a column between given dates and with given frequency
        :param column: (str) Label of the column
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param reverse_transform: (bool) Use reverse transformed data
        :param freq: (str) New data frequency('year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'microsecond')
        :param func: (func) Function to apply on data
        :return: (numpy.ndarray, numpy.ndarray, str) Data, scale and units
        """
        data, units = (self.reverse_trans(column), self.attributes[column]['units']) if reverse_transform else\
            (self.data[column], self.get_units(column))
        scale_column = self._get_scale(column)
        scale = self.data[scale_column]
        date_mask = self._preprocessor._date_mask(scale, from_date, to_date, self.attributes[scale_column]['timezone'])
        scale = np.array(scale)[date_mask]
        data = data[date_mask]

        if self._compatible_data([column]) and self._no_scale_in_data([column]):
            if freq:
                data, scale = self.downgrade_data_frequency(column, freq, from_date, to_date, func, reverse_transform)
                if freq == 'week':
                    years = set([s[0] for s in scale])
                    maxs = [max([s[1] for s in scale if s[0] == y]) + 1 for y in years]
                    dct = dict(zip(years, maxs))
                    scale = [i[0] + i[1] / dct[i[0]] for i in scale]
                scale = np.array(scale)
            else:
                scale = self._preprocessor._timestamps_to_dates(scale, tz=self.attributes[scale_column]['timezone'])
        return data, scale, units

    def plot(self, column, from_date=None, to_date=None, reverse_transform=True, axes=False, freq=None, func=np.sum):
        """
        Creates a plot to visualize data of a column in time
        :param column: (str) Label of the column
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param reverse_transform: (bool) Use reverse transformed data
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :param freq: (str) New data frequency('year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'microsecond')
        :param func: (func) Function to apply on data
        :return: (pyplot.axes) Axes of the plot
        """
        if not isinstance(from_date, type(None)):
            to_date = utils.calculate_to_date(to_date)
        data, scale, units = self._plot_calculate_data_scale_units(column, from_date, to_date,
                                                                   reverse_transform, freq, func)

        func_str = f"{str(func)[str(func).index('function ') + 9: str(func).index('at 0x') - 1]}"
        self._visualizer.plot(scale, data,
                              f'{column} - {func_str}({freq})' if freq else column, units, axes=axes)

    def hist(self, column, bins=10, reverse_transform=True, density=False, axes=False, plot_norm=False):
        """
        Creates a histogram-plot to visualize data contribution of a column
        :param column: (str) Label of the column
        :param bins: (int) Number of bins to divide contribution
        :param reverse_transform: (bool) Use reverse transformed data
        :param density: (bool) If True it creates probability density histogram
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :param plot_norm: (bool) If True and also density is True, it draws a normal distribution with same mean and
                                 standard deviation as these of the data.
        :return: (pyplot.axes) Axes of the plot
        """
        if self._compatible_data([column]):
            self._visualizer.hist(self.reverse_trans(column) if reverse_transform else self.data[column], column,
                                  self.attributes[column]['units'] if reverse_transform else self.get_units(column),
                                  bins, density, axes, plot_norm=plot_norm)

    def _compatible_data(self, columns):
        """
        Check if columns contain a series. If any of them contains a table, it raises an exception
        :param columns: (list(str)) List of the columns to be checked
        :return: True if every column are series
        """
        for column in columns:
            if self.data[column].dtype.names:
                raise TypeError(f"Incompatible data type in column '{column}'")
        return True

    def _get_scale(self, column):
        """
        Return scale column name of a column
        :param column: (str) Label of the column
        :return: (str) Label of the scale
        """
        scale = self.attributes[column]['scale']
        if not scale:
            raise TypeError(f"No scale found. Use attach_scale() function to set a scale for column '{column}'. ")
        return scale

    def _no_scale_in_data(self, columns):
        """
        Check if given columns not contain scales. Otherwise, it raises exception.
        :param columns: (list(str)) List of the columns to be checked
        :return: True if every column is not a scale
        """
        for column in columns:
            if self.is_scale(column):
                raise ValueError(f"Column '{column}' is a scale")
        return True

    def plot_seasons(self, column, period, from_date=None, to_date=None, reverse_transform=True, axes=None,
                     freq=None, func=np.sum):
        """
        Plot seasonal datas
        :param column: (str) Label of the column
        :param period: (str) Period of the plot ('annual', 'weekly', 'daily')
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param reverse_transform: (bool) Use reverse transformed data
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :param freq: (str) New data frequency('year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'microsecond')
        :param func: (func) Function to apply on data
        :return: (pyplot.axes) Axes of the plot
        """
        if not isinstance(from_date, type(None)):
            to_date = utils.calculate_to_date(to_date)
        legend = []
        valid_types = [('annual', 'hour'), ('annual', 'day'), ('annual', 'week'),
                       ('weekly', 'hour'), ('weekly', 'day'), ('daily', 'hour')]
        legend_format = {'annual': '%Y', 'weekly': '%d/%m/%y', 'daily': '%d/%m/%y %Hh'}
        if (period, freq) not in valid_types:
            raise TypeError(f'Period/Freq not valid values. Choose one of {valid_types}')

        data, scale, units = self._plot_calculate_data_scale_units(column, from_date, to_date,
                                                                   reverse_transform, freq, func)

        if period == 'annual':
            if freq == 'week':
                years = sorted(set([int(s) for s in scale]))
                data_splits = [data[scale.astype(np.int16) == year] for year in years]
                scale_splits = [scale[scale.astype(np.int16) == year] for year in years]
            else:
                years = sorted(set([s.year for s in scale]))
                indx = np.zeros(scale.shape, dtype=np.int64)
                for i in range(scale.shape[0]):
                    indx[i] = scale[i].year
                data_splits = [data[indx == year] for year in years]
                scale_splits = [scale[indx == year] for year in years]
        elif period == 'weekly':
            weeks = sorted(set([s.isocalendar()[:2] for s in scale]))
            indx = np.zeros(scale.shape, dtype=[('year', np.int16), ('week', np.int16)])
            for i in range(scale.shape[0]):
                indx[i] = scale[i].isocalendar()[:2]
            weeks = np.array(list(weeks), dtype=indx.dtype)
            data_splits = [data[indx == week] for week in weeks]
            scale_splits = [scale[indx == week] for week in weeks]
        else:
            days = sorted(set([(s.year, s.month, s.day) for s in scale]))
            indx = np.zeros(scale.shape, dtype=[('year', np.int16), ('month', np.int16), ('day', np.int16)])
            for i in range(scale.shape[0]):
                indx[i] = (scale[i].year, scale[i].month, scale[i].day)
            days = np.array(list(days), dtype=indx.dtype)
            data_splits = [data[indx == day] for day in days]
            scale_splits = [scale[indx == day] for day in days]

        for sc in scale_splits:
            if period == 'annual':
                try:
                    legend.append(f"{sc[0].strftime(legend_format[period])}")
                except AttributeError:
                    legend.append(f"{int(sc[0])}")
            else:
                legend.append(f"{sc[0].strftime(legend_format[period])} - {sc[-1].strftime(legend_format[period])}")

        return self._visualizer.plot_seasons(data_splits, name=f'{column} - ({period}/{freq})' if freq else column,
                                             units=units, legend_lines=legend, axes=axes)

    def plot_shapes(self, columns, from_date=None, to_date=None, axes=None, freq=None):
        """
        Compares plot-shapes of different columns
        :param columns: (list(str)) List of the columns to be plotted
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :param freq: (str) New data frequency ('year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'microsecond')
        :return: (pyplot.axes) Axes of the plot
        """
        if not isinstance(from_date, type(None)):
            to_date = utils.calculate_to_date(to_date)
        datas = []
        scale = None
        for column in columns:
            data, scale_, _ = self._plot_calculate_data_scale_units(column, from_date, to_date,
                                                                    True, freq, np.mean)
            datas.append(data)
            if isinstance(scale, type(None)):
                scale = scale_
            else:
                if not np.array_equal(scale, scale_):
                    raise ValueError('Scales not matching')

        return self._visualizer.plot_shapes(scale, datas, columns, axes)

    def _get_multi_differentiated_column(self, column, diffs):
        """
        Return multi-differentiated data from a column
        :param column: (str) Label of the column
        :param diffs: (list(int)) List of integers representing the differentiations to be calculated
        :return: (numpy.ndarray) multi-differentiated data
        """
        data = self.data[column].copy()
        for dif in diffs:
            data = self._preprocessor.differentiate(data, dif)
        return data

    def acf(self, column, nlags=None, qstat=False, alpha=None, missing='none', diffs=()):
        """
        Calculates autocorrelation function on multi-differentiate data of given column
        :param column: (str) Label of the column
        :param nlags: (int) Number of plotted lags
        :param qstat: (bool) If True returns the Ljung-Box q statistic for each autocorrelation coefficient
        :param alpha: (float) For calculation of confident intervals
        :param missing: (str) Specifying how the nans are treated ('none', 'raise', 'conservative', 'drop')
        :param diffs: (list(int)) List of integers representing the differentiations to be calculated
        :return: (list) Results from statsmodels module acf on multi-differentiate data of given column
        """
        data = self._get_multi_differentiated_column(column, diffs)
        return self._stats.acf(data, nlags=nlags, qstat=qstat, alpha=alpha, missing=missing)

    def pacf(self, column, nlags=None, method='yw', alpha=None, diffs=()):
        """
        Estimates partial autocorrelation on multi-differentiate data of given column
        :param column: (str) Label of the column
        :param nlags: (int) Number of plotted lags
        :param method: Method for calculations ('yw', 'ywm', 'ols-inefficient', 'ols-adjusted', 'ld', 'ldb', 'burg')
        :param alpha: (float) For calculation of confident intervals
        :param diffs: (list(int)) List of integers representing the differentiations to be calculated
        :return: (list) Results from statsmodels module pacf on multi-differentiate data of given column
        """
        data = self._get_multi_differentiated_column(column, diffs)
        return self._stats.pacf(data, nlags=nlags, method=method, alpha=alpha)

    def plot_acf(self, column, nlags=None, diffs=(), axes=None):
        """
        Create a figure with an ACF plot on multi-differentiate data of given column
        :param column: (str) Label of the column
        :param nlags: (int) Number of plotted lags
        :param diffs: (list(int)) List of integers representing the differentiations to be calculated
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        data = self._get_multi_differentiated_column(column, diffs)
        return self._visualizer.plot_acf(data, name=column, nlags=nlags, axes=axes)

    def plot_pacf(self, column, nlags=None, diffs=(), method='yw', axes=None):
        """
        Create a figure with a PACF plot on multi-differentiate data of given column
        :param column: (str) Label of the column
        :param nlags: (int) Number of plotted lags
        :param diffs: (list(int)) List of integers representing the differentiations to be calculated
        :param method: Specifies which method for the calculations to use
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        data = self._get_multi_differentiated_column(column, diffs)
        if diffs:
            while np.isnan(data[0]):
                data = data[1:]
        return self._visualizer.plot_pacf(data, name=column, nlags=nlags, method=method, axes=axes)

    def plot_moving_averages(self, column, period, axes=None):
        """
        Plot Moving Averages data for the given column
        :param column: (str) Label of the column
        :param period: (int) Period of the Moving Averages
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        scale = self._get_scale_strings(column)
        return self._visualizer.plot_moving_averages(scale, self.data[column], name=f'{column} - period {period}',
                                                     period=period, units=self.attributes[column]['units'], axes=axes)

    def plot_seasonality(self, column, period, trend_sign='sub', number_of_periods=1, axes=None):
        """
        Plot seasonality of the given data column with application of classical decomposition
        :param column: (str) Label of the column
        :param period: (int) Seasonality period
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :param number_of_periods: (int) number of periods to plot (None to plot all the data)
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        scale = self._get_scale_strings(column)
        return self._visualizer.plot_seasonality(scale, self.data[column], name=f'{column} - period {period}',
                                                 number_of_periods=number_of_periods, trend_sign=trend_sign,
                                                 period=period, units=self.attributes[column]['units'], axes=axes)

    def _get_scale_strings(self, column):
        """
        Get scale data as date-strings
        :param column: (str) name of the scale column
        :return: (numpy.ndarray) date-strings
        """
        scale = self.attributes[column]['scale']
        timezone = self.attributes[scale]['timezone']
        return utils.timestamp_to_date_str(self.data[scale], timezone)

    def plot_classical_decomposition(self, column, period, number_of_periods=1, trend_sign='div', seasonal_sign='div'):
        """
        Plot data, trend, seasonality and residuals using classical decomposition method on a data column
        :param column: (str) Label of the column
        :param period: (int) Seasonal period
        :param number_of_periods: (int) number of periods to plot (None to plot all the data)
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :param seasonal_sign: (str) 'div' for multiplicative seasonality, 'sub' for additive seasonality
        :return: (None)
        """
        scale = self._get_scale_strings(column)
        self._visualizer.plot_classical_decomposition(self.data[column], scale, name=f'{column} - period {period}',
                                                      number_of_periods=number_of_periods, trend_sign=trend_sign,
                                                      seasonal_sign=seasonal_sign, period=period,
                                                      units=self.attributes[column]['units'])

    def data_summary(self, columns=None):
        """
        Returns a summary for the data
        :param columns: (list(str)) list of the column names
        :return: (str) printable summary of data
        """
        return self._EF.data_controller.data_summary(self.name, columns)

