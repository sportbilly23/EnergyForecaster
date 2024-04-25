import numpy as np
import pytz
from scipy.stats import boxcox
import datetime
from utils import calculate_to_date
DATE_FORMATS = """\n    %a				Abbreviated weekday name.									Sun, Mon, ...
    %A				Full weekday name.											Sunday, Monday, ...
    %w				Weekday as a decimal number.								0, 1, ..., 6
    %d				Day of the month as a zero-padded decimal.					01, 02, ..., 31
    %-d				Day of the month as a decimal number.						1, 2, ..., 30
    %b				Abbreviated month name.										Jan, Feb, ..., Dec
    %B				Full month name.											January, February, ...
    %m				Month as a zero-padded decimal number.						01, 02, ..., 12
    %-m				Month as a decimal number.									1, 2, ..., 12
    %y				Year without century as a zero-padded decimal number.		00, 01, ..., 99
    %-y				Year without century as a decimal number.					0, 1, ..., 99
    %Y				Year with century as a decimal number.						2013, 2019 etc.
    %H				Hour (24-hour clock) as a zero-padded decimal number.		00, 01, ..., 23
    %-H				Hour (24-hour clock) as a decimal number.					0, 1, ..., 23
    %I				Hour (12-hour clock) as a zero-padded decimal number.		01, 02, ..., 12
    %-I				Hour (12-hour clock) as a decimal number.					1, 2, ... 12
    %p				Locale’s AM or PM.											AM, PM
    %M				Minute as a zero-padded decimal number.						00, 01, ..., 59
    %-M				Minute as a decimal number.									0, 1, ..., 59
    %S				Second as a zero-padded decimal number.						00, 01, ..., 59
    %-S				Second as a decimal number.									0, 1, ..., 59
    %f				Microsecond as a decimal number, zero-padded on the left.	000000 - 999999
    %z				UTC offset in the form +HHMM or -HHMM.
    %Z				Time zone name.
    %j				Day of the year as a zero-padded decimal number.			001, 002, ..., 366
    %-j				Day of the year as a decimal number.			 		 	1, 2, ..., 366
    %U				Week number of the year (Sunday as the first day
                    of the week). All days in a new year preceding the
                    first Sunday are considered to be in week 0.				00, 01, ..., 53
    %W				Week number of the year (Monday as the first day 
                    of the week). All days in a new year preceding the
                    first Monday are considered to be in week 0.				00, 01, ..., 53
    %c				Locale’s appropriate date and time representation.			Mon Sep 30 07:06:05 2013
    %x				Locale’s appropriate date representation.					09/30/13
    %X				Locale’s appropriate time representation.					07:06:05
    %%				A literal '%' character.									%"""
WRONG_MODE_ERROR = 'Mode selection must be one of "var", "cos-sin" and "one-hot"'
WEEKDAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')


class Preprocessor:
    """
    Transforms/Normalizes/Extends datasets for Energy Forecaster
    """
    def __init__(self, ef):
        self._EF = ef

    def log(self, base, data):
        """
        Data logarithmic transformation
        :param base: (int) Base of the logarithm transformation
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.emath.logn(base, data), (lambda x: np.emath.logn(base, x), lambda x: np.power(10, x))

    def log2(self, data):
        """
        Data binary logarithmic transformation (base 2)
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.log2(data), (lambda x: np.log2(x), lambda x: np.power(2, x))

    def log10(self, data):
        """
        Data common logarithmic transformation (base 10)
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.log10(data), (lambda x: np.log10(x), lambda x: np.power(2, x))

    def ln(self, data):
        """
        Data natural logarithmic transformation (base e)
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.log(data), (lambda x: np.log(x), lambda x: np.exp(x))

    def exp(self, data):
        """
        Data natural exponential transformation (base e)
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.exp(data), (lambda x: np.exp(x), lambda x: np.log(x))

    def exp2(self, data):
        """
        Data binary exponential transformation (base 2)
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.exp2(data), (lambda x: np.exp2(x), lambda x: np.log2(x))

    def boxcox(self, data, lamda=None):
        """
        Box-Cox transformation with lambda parameter
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        bc = boxcox(data, lmbda=lamda)
        if isinstance(lamda, type(None)):
            bc, lamda = bc

        return boxcox(data, lmbda=lamda), (lambda x: boxcox(x, lamda=lamda), lambda x: (x * lamda + 1) ** (1 / lamda))

    def limit_output(self, data, low, hi):
        """
        Setting limits to the output
        :param data: (numpy.ndarray) Output data to be transformed
        :param low: (float) The lower desired value of the output
        :param hi: (float) The higher desired value of the output
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        return np.log((data - low)/(hi - data)), (lambda x: np.log((x - low)/(hi - x)),
                                                  lambda x: (hi - low) * np.exp(x) / (1 + np.exp(x)) + low)

    def minmax(self, data):
        """
        Min-Max normalization
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        mn = np.nanmin(data)
        mx = np.nanmax(data)
        return (data - mn) / (mx - mn), (lambda x: (x - mn) / (mx - mn), lambda x: x * (mx - mn) + mn)

    def standard(self, data, centered=True, devarianced=True):
        """
        Standard normalization
        :param data: (numpy.ndarray) Data to be transformed
        :param centered: (bool) Subtracks mean value to center transformed data on axes
        :param devarianced: (bool) Divides by standard deviation to de-variance transformed data
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        mn = np.nanmean(data) if centered else 0
        std = np.nanstd(data) if devarianced else 1
        return (data - mn) / std, (lambda x: (x - mn) / std, lambda x: x * std + mn)

    def robust(self, data, centered=True, quantile_range=(.25, .5)):
        """
        Robust normalization
        :param data: (numpy.ndarray) Data to be transformed
        :param centered: (bool) Subtracks median value to center transformed data on axes
        :param quantile_range: (tuple(float)) Defines quantiles to de-variance transformed data
        :return: (numpy.ndarray, func) Transformed data and reverse transformation function
        """
        quantile_range = sorted(quantile_range)
        md = np.quantile(data, q=0.5) if centered else 0
        if isinstance(quantile_range, type(None)):
            q = 1
        else:
            q = np.quantile(data, q=quantile_range[1]) - np.quantile(data, q=quantile_range[0])

        return (data - md) / q, (lambda x: (x - md) / q, lambda x: x * q + md)

    def differentiate(self, data, period=1):
        """
        Differentiate data
        :param data: (numpy.ndarray) Data to be transformed
        :param period: (int) Period of the differantiation
        :return: (numpy.ndarray) Transformed data
        """
        return np.array([np.nan] * period + [1] * (len(data) - period)) * np.roll(data, period) - data

    def croston_method(self, data):
        """
        Transform sparse series to inter-arrival/quantity series
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray) Transformed data
        """
        inter_arrival = []
        quantity = []
        zeros = 0
        for d in data:
            if d == 0:
                zeros += 1
            else:
                inter_arrival.append(zeros + 1)
                quantity.append(d)
                zeros = 0

        croston = np.zeros(shape=(len(quantity),), dtype=[('inter_arrival', np.int32), ('quantity', np.int32)])
        croston['inter_arrival'], croston['quantity'] = inter_arrival, quantity
        return croston

    def to_timestamp(self, data, form):
        """
        Transforms strings of dates to timestamps
        :param data:(numpy.ndarray) Data to be transformed
        :param form: (str) Format of string data
        :return: (numpy.ndarray) Transformed data - Timestamps
        """
        try:
            dt = [datetime.datetime.strptime(d, form) for d in data]
            if dt[0].tzinfo:
                d_utc = [d.astimezone(pytz.timezone('UTC')) for d in dt]
            else:
                d_utc = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second,
                                           d.microsecond, pytz.utc) for d in dt]
            deltas = [(d_utc[i + 1] - d_utc[i]).seconds for i in range(len(d_utc) - 1)]
            if len(set(deltas)) > 1:
                raise Exception('Converting to UTC created non-periodic intervals')
        except ValueError:
            raise ValueError('\n'.join(['Wrong date format', DATE_FORMATS]))

        return np.array([datetime.datetime.timestamp(d) for d in d_utc], dtype=np.float64)

    def _create_indicators(self, data, func, dtype=np.uint8):
        """
        Creates indicators for data using a given function. Function used to catch errors in time zones
        :param data: (numpy.ndarray) Data to be transformed
        :param func: (func) Function to make the transformation
        :param dtype: (numpy.dtype) dtype of the returned indicators
        :return: (numpy.ndarray) Data transformed as indicators
        """
        try:
            indicators = np.array([func(d) for d in data], dtype=dtype)
        except pytz.exceptions.UnknownTimeZoneError:
            raise pytz.exceptions.UnknownTimeZoneError(f'Valid time zones:\n{pytz.all_timezones}')
        return indicators

    def _create_dummies_array(self, name, number, indicators):
        """
        Create one-hot encoding from a given 1d array of indicators
        :param name: (str) To name the columns of the table
        :param number: (int) Number of different indicators
        :param indicators: (numpy.ndarray) 1d array of indicators
        :return: (numpy.array) Table with one-hot encoded indicators
        """
        arr = np.zeros(shape=(indicators.shape[0],),
                       dtype=list(zip([f'{name}_{d}' for d in range(1, number)], [np.uint8] * (number - 1))))
        for d in range(1, number):
            arr[f'{name}_{d}'] = (indicators == d).astype(np.uint8)
        return arr

    def _time_zone(self, time_zone):
        """
        Returns pytz.timezone from timezone-string
        :param time_zone: (str) timezone string
        :return: (pytz.timezone) corresponding pytz timezone
        """
        return time_zone if time_zone.__class__.__module__ == 'pytz' else pytz.timezone(time_zone)

    def weekend(self, data, time_zone):
        """
        Weekday/weekend indicator
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :return: (numpy.ndarray) Weekend indicators
        """
        return self._create_indicators(
            data, lambda x: datetime.datetime.fromtimestamp(x, tz=self._time_zone(time_zone)).weekday() > 4)

    def _cos_sin_transformation(self, name, indicators):
        """
        Cyclic transformation of indicators (cos/sin)
        :param name: (str)  To name the columns of the table
        :param indicators: (numpy.ndarray) 1d array of indicators
        :return: (numpy.array) Table with cyclic transformed indicators
        """
        mn = np.nanmin(indicators)
        mx = np.nanmax(indicators)

        cos_sin = np.zeros(shape=(indicators.shape[0],), dtype=[(f'{name}_sin', np.float64),
                                                                (f'{name}_cos', np.float64)])
        cos_sin[f'{name}_sin'] = np.sin(2 * np.pi * (indicators - mn) / (mx - mn))
        cos_sin[f'{name}_cos'] = np.cos(2 * np.pi * (indicators - mn) / (mx - mn))
        return cos_sin

    def weekday(self, data, time_zone, mode='one-hot'):
        """
        Weekday indicators/one-hot encoding/cyclical encoding
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :return: (numpy.array) Weekday indicators
        """
        indicators = self._create_indicators(
            data, lambda x: datetime.datetime.fromtimestamp(x, tz=self._time_zone(time_zone)).weekday())
        if mode == 'one-hot':
            return self._create_dummies_array('weekday', 7, indicators)
        elif mode == 'var':
            return indicators
        elif mode == 'cos-sin':
            return self._cos_sin_transformation('weekday', indicators)
        else:
            raise KeyError(WRONG_MODE_ERROR)

    def monthday(self, data, time_zone, mode='one-hot'):
        """
        Monthday indicators/one-hot encoding/cyclical encoding
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :return: (numpy.array) Monthday indicators
        """
        indicators = self._create_indicators(
            data, lambda x: datetime.datetime.fromtimestamp(x, tz=self._time_zone(time_zone)).day)
        if mode == 'one-hot':
            return self._create_dummies_array('monthday', 31, indicators)
        elif mode == 'var':
            return indicators
        elif mode == 'cos-sin':
            return self._cos_sin_transformation('monthday', indicators)
        else:
            raise KeyError(WRONG_MODE_ERROR)

    def day_hour(self, data, time_zone, mode='one-hot'):
        """
        Day hour indicators/one-hot encoding/cyclical encoding
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :return: (numpy.array) Monthday indicators
        """
        indicators = self._create_indicators(
            data, lambda x: datetime.datetime.fromtimestamp(x, tz=self._time_zone(time_zone)).hour)
        if mode == 'one-hot':
            return self._create_dummies_array('day_hour', 24, indicators)
        elif mode == 'var':
            return indicators
        elif mode == 'cos-sin':
            return self._cos_sin_transformation('day_hour', indicators)
        else:
            raise KeyError(WRONG_MODE_ERROR)

    def _calculate_year_day(self, x, time_zone):
        """
        Calculates day of the year for a timestamp
        :param x: (float) Timestamp
        :param time_zone: (str) Time zone label
        :return: (int) Day of the year
        """
        to_ = datetime.datetime.fromtimestamp(x).astimezone(self._time_zone(time_zone))
        from_ = datetime.datetime(to_.year, 1, 1, 0, 0, 0, 0, to_.tzinfo)
        return (to_ - from_).days + 1
    
    def year_day(self, data, time_zone, mode='one-hot'):
        """
        Day of year indicators/one-hot encoding/cyclical encoding
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :return: (numpy.array) Day of year indicators
        """
        indicators = self._create_indicators(data, lambda x: self._calculate_year_day(x, time_zone), np.uint16)
        if mode == 'one-hot':
            return self._create_dummies_array('year_day', 365, indicators)
        elif mode == 'var':
            return indicators
        elif mode == 'cos-sin':
            return self._cos_sin_transformation('year_day', indicators)
        else:
            raise KeyError(WRONG_MODE_ERROR)

    def year_week(self, data, time_zone, mode='one-hot'):
        """
        Week of year indicators/one-hot encoding/cyclical encoding
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :return: (numpy.array) Week of year indicators
        """
        days = self.weekday(data, time_zone, mode='var')
        prev_week = week = prev_year = -1
        weeks = []
        for day_i, date_j in zip(days, data):
            current_year = datetime.datetime.fromtimestamp(date_j).astimezone(self._time_zone(time_zone)).year
            if current_year != prev_year:
                prev_year = current_year
                week = 1
                prev_week = day_i
            if day_i == 0 and prev_week == 6:
                week += 1
            weeks.append(week)
            prev_week = day_i
        indicators = np.array(weeks)
        if mode == 'one-hot':
            return self._create_dummies_array('year_week', 54, indicators)
        elif mode == 'var':
            return indicators
        elif mode == 'cos-sin':
            return self._cos_sin_transformation('year_week', indicators)
        else:
            raise KeyError(WRONG_MODE_ERROR)

    def year_month(self, data, time_zone, mode='one-hot'):
        """
        Month indicators/one-hot encoding/cyclical encoding
        :param data: (numpy.ndarray) Data to be transformed
        :param time_zone: (str) Time zone label
        :param mode: (str) "var"/"one-hot"/"cos-sin"
        :return: (numpy.array) Month indicators
        """
        indicators = self._create_indicators(
            data, lambda x: datetime.datetime.fromtimestamp(x, tz=self._time_zone(time_zone)).month)
        if mode == 'one-hot':
            return self._create_dummies_array('year_month', 12, indicators)
        elif mode == 'var':
            return indicators
        elif mode == 'cos-sin':
            return self._cos_sin_transformation('year_month', indicators)
        else:
            raise KeyError(WRONG_MODE_ERROR)

    def _count_month_weekdays(self, start):
        """
        Counts the number of the seven weedays for a given month (starts to count from given date)
        :param start: (datetime) A date from when it starts to count (select the 1st day of the month to count month's
                                 weekdays as a whole)
        :return: (tuple(int)) Number of the seven differnt weekdays in a month (from given date)
        """
        current_month = month = start.month
        weekdays = [0, 0, 0, 0, 0, 0, 0]
        while current_month == month:
            weekdays[start.weekday()] += 1
            start += datetime.timedelta(days=1)
            current_month = start.month
        return tuple(weekdays)

    def month_weekdays(self, from_year_month, to_year_month):
        """
        Creates monthly numbers of weekdays between two dates
        :param from_year_month: (tuple(int, int)) Starting year and month
        :param to_year_month:  (tuple(int, int)) Ending year and month
        :return: (numpy.ndarray) Monthly numbers of weekdays between two dates
        """
        from_year, from_month = from_year_month
        to_year, to_month = to_year_month
        all_months = (to_year - from_year - 1) * 12 + 13 - from_month + to_month
        weekdays = np.zeros(shape=(all_months,), dtype=[(WEEKDAYS[i], np.uint8) for i in range(7)])
        month = 0
        for current_year in range(from_year, to_year + 1):
            for current_month in range(from_month if current_year == from_year else 1,
                                       to_month + 1 if current_year == to_year else 13):
                weekdays[month] = self._count_month_weekdays(datetime.datetime(current_year, current_month, 1))
                month += 1
        return weekdays

    def _is_in_dates(self, x, dates, time_zone):
        """
        Checks if a date x exists in a list of dates
        :param x: (datetime) The given date
        :param dates: list(tuples) List of tuples with dates in format (int, int, int) for year, month and day
        :param time_zone: (str) Time zone label
        :return: (int) 1 for matching / 0 otherwise
        """
        dt = datetime.datetime.fromtimestamp(x).astimezone(self._time_zone(time_zone))
        return int((dt.year, dt.month, dt.day) in dates)

    def public_holidays(self, data, holidays, time_zone):
        """
        Creates one-hot encoding for given list of holidays
        :param data: (numpy.ndarray) Data to be transformed
        :param holidays: (list(tuple)) List of tuples containing year, month and monthday of holidays
        :param time_zone: (str) Time zone label
        :return: (numpy.ndarray) One-hot encoding for national holidays
        """
        return self._create_indicators(
            data, lambda x: self._is_in_dates(x, holidays, time_zone))

    def lagged_series(self, data, name, lags=(1,)):
        """
        Creates a number of lagged series for the input data
        :param data: (numpy.ndarray) Data to be transformed
        :param name: (str) To name the columns of lagged series
        :param lags: (tuple) Tuple with lags to be created
        :return: (numpy.ndarray) Lagged series of the data
        """
        lags = sorted(lags)
        lag_series = np.zeros(shape=(data.shape[0],), dtype=[(f'{name}_lag-{lag}', data.dtype) for lag in lags])
        for lag in lags:
            lag_series[f'{name}_lag-{lag}'] = np.roll(data, lag)
            try:
                lag_series[f'{name}_lag-{lag}'][:lag] = ''
            except ValueError:
                try:
                    lag_series[f'{name}_lag-{lag}'][:lag] = np.nan
                except ValueError:
                    lag_series[f'{name}_lag-{lag}'][:lag] = -1
        return lag_series

    def fill_backward(self, data):
        """
        Fills nan values with next value
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray) Data with filled the nan values (except last line's nans)
        """
        nans = np.argwhere(np.isnan(data))
        new_data = data.copy()
        for i in reversed(nans):
            try:
                new_data[i] = new_data[i + 1]
            except IndexError:
                continue
        return new_data

    def fill_forward(self, data):
        """
        Fills nan values with previous value
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray) Data with filled the nan values (except first line's nans)
        """
        nans = np.argwhere(np.isnan(data))
        new_data = data.copy()
        for i in nans:
            if i > 0:
                new_data[i] = new_data[i - 1]
        return new_data

    def fill_linear(self, data):
        """
        Fills nan values with linear interpolation between previous and next values
        :param data: (numpy.ndarray) Data to be transformed
        :return: (numpy.ndarray) Data with filled the nan values
        """
        nans = np.argwhere(np.isnan(data))
        new_data = data.copy()
        for i in nans:
            int_i = int(i)
            if np.isnan(new_data[i]):
                next_i = int_i + 1
                try:
                    while np.isnan(new_data[next_i]):
                        next_i += 1
                except IndexError:
                    next_i = -1
                if i > 0:
                    prev_i = int_i - 1
                else:
                    prev_i = -1

                if prev_i == next_i == -1:
                    raise ValueError('There is a full nan column')
                elif prev_i == -1:
                    ii = range(0, next_i)
                    values = np.linspace(new_data[next_i], new_data[next_i], next_i + 2)[1:-1]
                elif next_i == -1:
                    ii = range(int_i, data.shape[0])
                    values = np.linspace(new_data[prev_i], new_data[prev_i], data.shape[0] - int_i + 2)[1:-1]
                else:
                    ii = range(int_i, next_i + 1)
                    values = np.linspace(new_data[prev_i], new_data[next_i], next_i - int_i + 2)[1:-1]
                for ci, value in zip(ii, values):
                    new_data[ci] = value
        return new_data

    def _date_mask(self, scale, from_date, to_date, tz):
        """
        Create a boolean series (smart index) to use it for time interval selection
        :param scale: (numpy.ndarray) Time scale for the creation of smart index
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param tz: (pytz.timezone) Timezone of the scale
        :return: (list) Smart index for given time interval selection
        """
        try:
            to_date = calculate_to_date(to_date)
            ln = len(from_date)
            if ln < 3:
                from_date = tuple(1 if i >= ln else from_date[i] for i in range(3))
            from_date_ = tz.localize(datetime.datetime(*from_date)).timestamp()
            to_date_ = tz.localize(datetime.datetime(*to_date)).timestamp()
            ln = len(scale)
            scale_tf_1 = [s >= from_date_ for s in scale] if from_date_ else [True] * ln
            scale_tf_2 = [s <= to_date_ for s in scale] if to_date_ else [True] * ln
        except TypeError:
            return [True] * len(scale)
        return [i & j for i, j in zip(scale_tf_1, scale_tf_2)]

    def _timestamps_to_dates(self, timestamps, tz=None, freq='microsecond'):
        """
        Creates datetimes from timestamps with respect of the selected frequency
        :param timestamps: (numpy.ndarray) Timestamps
        :param tz: (pytz.timezone) Timezone
        :param freq: (str) The selected frequency to categorize the timestamps
        :return: (numpy.ndarray) Categorized datetimes with respect of given frequency
        """
        periods = {'year': 0, 'month': 1, 'week': 2, 'day': 3, 'hour': 4, 'minute': 5, 'second': 6, 'microsecond': 7}
        dates = []

        for s in timestamps:
            s = datetime.datetime.fromtimestamp(s, tz=tz)
            dates.append(datetime.datetime(s.year,
                                           s.month if periods[freq] > 0 else 1,
                                           s.day if periods[freq] > 1 else 1,
                                           s.hour if periods[freq] > 3 else 0,
                                           s.minute if periods[freq] > 4 else 0,
                                           s.second if periods[freq] > 5 else 0,
                                           s.microsecond if periods[freq] > 6 else 0,
                                           s.tzinfo))

        if freq == 'week':
            for i, date in enumerate(dates):
                isocal = date.isocalendar()[:2]
                dates[i] = isocal if date.year == isocal[0] else (date.year, 0)
            dates = np.array(dates, dtype=[('year', np.int16), ('week', np.int16)])
        else:
            dates = np.array(dates)
        return dates

    def downgrade_data_frequency(self, data, scale, freq, from_date, to_date, tz=None, func=np.sum):
        """
        Change frequency of data in a given time interval
        :param data: (numpy.ndarray) Data to be used
        :param scale: (numpy.ndarray) Time scale of the given data
        :param freq: (str) New data frequency ('year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'microsecond')
        :param from_date: (list(int)) Starting date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param to_date: (list(int)) Ending date as a list (Year, Month, Day, Hour, Minute, Second, Microsecond)
        :param tz: (pytz.timezone) Timezone
        :param func: (func) Function to use in categorized data
        :return: (numpy.ndarray, numpy.ndarray) categorized data and new scale
        """
        lmbd = lambda x: func(x)

        date_mask = self._date_mask(scale, from_date, to_date, tz)

        scale = scale[date_mask]
        data = data[date_mask]

        dates = self._timestamps_to_dates(scale, tz, freq)
        set_dates = sorted(set(dates.tolist()))

        if freq == 'week':
            return np.array([lmbd(data[dates == d]) for d in np.array(set_dates, dtype=dates.dtype)]), set_dates
        else:
            return np.array([lmbd(data[dates == d]) for d in set_dates]), set_dates

    def reverse_trans(self, data, trans_list):
        """
        Reverse transformation of data using a transformation list
        :param data: (numpy.ndarray) Data to be back-transformed
        :param trans_list: (list(dict)) Transformation list (with dicts {'func': func(), 'rev': rev()})
        :return: (numpy.ndarray) Back-transformed data
        """
        rev_data = data.copy()
        for trans in trans_list:
            rev_data = trans['rev'](rev_data)

        return rev_data
