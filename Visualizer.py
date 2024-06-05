import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Creates graphical visualizations for Energy Forecaster
    """
    def __init__(self, ef):
        self._EF = ef

    def _limit_xticks(self, data, axes, number_of_ticks=20, rotation=30, period=1, grid=False, text=True):
        """
        Transformation and limitation of xticks for better view
        :param data: (numpy.ndarray/list) The data to plot
        :param axes: (pyplot.axes) The axes where the plot taking place
        :param number_of_ticks: (int) Maximum number of x ticks
        :param rotation: (int) Rotation angle for numeration
        :param period: (int) Period of the plot
        :param grid: (bool) True to show grid
        :param text: (bool) True to use data/text or False to use count of periods
        :return: (None)
        """
        plt.sca(axes)
        ln = len(data)
        tick_window = period
        count = 1
        while ln / tick_window > number_of_ticks:
            count += 1
            tick_window = period * count
        plt.xticks(range(0, ln - tick_window // 2, tick_window),
                   [data[i * count * period] for i in range(round(ln / tick_window))] if text else\
                   [i * count for i in range(round(ln / tick_window))], rotation=rotation)
        plt.grid(grid)

    def _get_next_figure_name(self, pattern):
        """
        Check for open figure windows with same name to get next valid name
        :param pattern: (str) The pattern name
        :return: (str) Next valid name
        """
        mx = 0
        for fig in plt.get_fignums():
            try:
                name, number = plt.figure(fig)._label.split(' - ')
            except ValueError:
                continue
            if name == pattern:
                mx = max(mx, int(number))
        return f'{pattern} - {mx + 1}'

    def plot_acf(self, data, name='data', nlags=None, axes=None):
        """
        Create a figure with an ACF plot
        :param data: (numpy.ndarray) Data to create the plot
        :param name: (str) The name of the plot-title
        :param nlags: (int) Number of plotted lags
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('ACF Plot'))
            axes = figure.subplots()
        acf = self._EF.data_statistics.acf(data, nlags=nlags, missing='conservative')
        rng = np.arange(1, len(acf))
        plt.hlines(0, rng[0], rng[-1], color='grey')
        axes.bar(rng, acf[1:], width=.2)
        axes.plot(rng, [-1.96 / len(data) ** 0.5] * len(rng), color='red', linestyle='dashed')
        axes.plot(rng, [1.96 / len(data) ** 0.5] * len(rng), color='red', linestyle='dashed')
        axes.set_title(f'{name}')
        axes.set_xlabel('lag')
        plt.xticks(rng)
        axes.set_ylabel('correlation')
        return axes

    def plot_pacf(self, data, name='data', nlags=None, method='yw', axes=None):
        """
        Create a figure with a PACF plot
        :param data: (numpy.ndarray) Data to create the plot
        :param name: (str) The name of the plot-title
        :param nlags: (int) Number of plotted lags
        :param method: Specifies which method for the calculations to use
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('PACF Plot'))
            axes = figure.subplots()
        pacf = self._EF.data_statistics.pacf(data, nlags=nlags, method=method)

        rng = np.arange(1, len(pacf))
        plt.hlines(0, rng[0], rng[-1], color='grey')
        axes.bar(rng, pacf[1:], width=.2)
        axes.plot(rng, [-1.96 / len(data) ** 0.5] * len(rng), color='red', linestyle='dashed')
        axes.plot(rng, [1.96 / len(data) ** 0.5] * len(rng), color='red', linestyle='dashed')
        axes.set_title(f'{name}')
        axes.set_xlabel('lag')
        plt.xticks(rng)
        axes.set_ylabel('partial correlation')
        return axes

    def hist(self, data, name='data', units='units', bins=10, density=False, axes=None, plot_norm=False):
        """
        Creates a histogram-plot to visualize data contribution of a column
        :param data: (numpy.ndarray) Data of the variable to plot
        :param name: (str) Names of the variable
        :param units: (str) Units of the variable
        :param bins: (int) Number of histogram bins
        :param density: (bool) If True it creates probability density histogram
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :param plot_norm: (bool) If True and also density is True, it draws a normal distribution with same mean and
                                 standard deviation as these of the data.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Histogram'))
            axes = figure.subplots()
        axes.hist(data, bins=bins, histtype='stepfilled', density=density)
        axes.set_title(f'{name} - ({bins} bins)')
        axes.set_xlabel(f'bins({units})')
        axes.set_ylabel('counts')

        if plot_norm and density:
            std = np.std(data, ddof=2)
            mu = np.mean(data)
            rng = np.linspace(mu - 4 * std, mu + 4 * std, 1000)
            nd_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * (1 / std * (rng - mu)) ** 2)
            plt.plot(rng, nd_y)

        return axes

    def plot(self, scale, data, name='data', units='units', xlab='time', axes=None):
        """
        Creates a plot to visualize variable into time
        :param scale: (numpy.ndarray) Scale data of the variable to plot
        :param data: (numpy.ndarray) Data of the variable to plot
        :param name: (str) Name for title
        :param units: (str) Units of the variable
        :param xlab: (str) x label
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Plot'))
            axes = figure.subplots()
        axes.plot(scale, data)
        axes.set_title(f'{name}')
        axes.set_xlabel(xlab)
        axes.set_ylabel(units)
        return axes

    def plot_residuals(self, scale, resids, name='residuals', units='units', axes=None):
        """
        Plot residuals
        :param scale: (numpy.ndarray) The scale data
        :param resids: (numpy.ndarray) The residuals
        :param name: (str) Name for title
        :param units: (str) Units of the residuals
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Residuals'))
            axes = figure.subplots()
        self._limit_xticks(scale, axes, number_of_ticks=20, rotation=0, text=True, grid=True)
        axes.bar(scale, resids, width=.2)

        axes.set_title(f'{name}')
        axes.set_xlabel('time')
        axes.set_ylabel(units)

        return axes


class VisualizeData(Visualizer):
    """
    Visualizations for data
    """
    COMPARE_PERIODS_GRAPH_TYPE = {0: 'Annual/Mean(Day)', 1: 'Annual/Mean(Week)', 2: 'Annual/Mean(Month)',
                                  3: 'Weekly/Mean(Day)', 4: 'Weekly/Mean(Hour)', 5: 'Daily/Mean(Hour)'}

    def scatter(self, data_1, data_2, names=('data_1', 'data_2'), units=('units', 'units'),
                marker='o', markersize=2, axes=None):
        """
        Creates scatter-plot to visualize correlation between variables
        :param data_1: (numpy.ndarray) Data of the first variable (x-axes)
        :param data_2: (numpy.ndarray) Data of the second variable (y-axes)
        :param names: (tuple(str)) Names of the variables
        :param units: (tuples(str)) Units of the variables
        :param marker: (str) Marker of the plot
        :param markersize: (int) Size of the marker
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Scatter Plot'))
            axes = figure.subplots()
        axes.plot(data_1, data_2, linewidth=0.0, marker=marker, markersize=markersize)
        axes.set_title(f'{names[0]} vs {names[1]}')
        axes.set_xlabel(units[0])
        axes.set_ylabel(units[1])
        return axes

    def plot_shapes(self, scale, datas, names, axes=None):
        """
        Compares plot-shapes of different variables
        :param scale: (numpy.ndarray) Scale data of the variable to plot
        :param datas: (list(numpy.ndarray)) Datas of the variables to plot
        :param names: (list(str)) Names of the variables
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        legend = [] if not axes else [t._text for t in axes.get_legend().texts]
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Shapes Plot'))
            axes = figure.subplots()
        self._limit_xticks(scale, axes, grid=True)
        title = ' vs '.join(names)
        for data, name in zip(datas, names):
            data = self._EF.preprocessor.minmax(data)[0]
            axes = self.plot(scale, data, name=title, units='', axes=axes)
            legend.append(name)

        axes.legend(labels=legend)

    def plot_seasons(self, datas, legend_lines, name='data', units='units', axes=None):
        """
        Plot seasonal datas
        :param datas: (list(numpy.ndarray)) List of seasonal divided datas
        :param name: (str) Name of variable to plot
        :param units: (str) Units of variable to plot
        :param legend_lines: (list(str)) Strings for legend additions (dates for seasons)
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        legend = legend_lines if not axes else [t._text for t in axes.get_legend().texts] + legend_lines
        for dt in datas:
            axes = self.plot(1 + np.arange(len(dt)), dt, name, units, axes=axes)

        axes.legend(labels=legend)

        return axes

    def _moving_averages(self, data, period, centering=True):
        """
        Returns moving averages data for a defined period
        :param data: (numpy.ndarray) Data to extract moving averages
        :param period: (int) Period of the moving averages
        :param centering: (bool) True to execute a 2-MA after an even period MA to center data
        :return: (numpy.ndarray) Moving average data
        """
        ln = len(data)
        new_data = np.array([np.mean(data[i:period + i]) for i in range(ln - period + 1)])
        if centering and period % 2 == 0:
            new_data = self._moving_averages(new_data, 2, False)
        return new_data

    def _moving_average_data_offset(self, length, period):
        """
        Returns a function to use it for selecting the right slice of source with the created Moving Averages
        :param length: (int) length of the source data
        :param period: (int) period of MA
        :return: (func) Function to create sliced data
        """
        offset = period // 2
        return lambda x: x[offset: length - offset]

    def _get_trend_rest_data(self, data, period, trend_sign):
        """
        Separate trend data from the rest using classical decomposition (Moving Averages)
        :param data: (numpy.ndarray) Data to decomposite
        :param period: (int) Period of Moving Averages
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :return: (tuple(numpy.ndarray)) trend data and rest of the data
        """
        trend = self._moving_averages(data, period)
        ln = len(data)
        if trend_sign == 'div':
            return trend, self._moving_average_data_offset(ln, period)(data) / trend
        elif trend_sign == 'sub':
            return trend, self._moving_average_data_offset(ln, period)(data) - trend

    def _get_seasonality(self, detrend_data, period):
        """
        Get seasonality from the de-trended data using classical decomposition
        :param detrend_data: (numpy.ndarray) De-trended data
        :param period: (int) period of seasonality
        :return: (numpy.ndarray) seasonal data
        """
        ln = len(detrend_data)
        seasonal_means = np.nanmean(np.pad(detrend_data, (0, int(np.ceil(ln / period) * period - ln)),
                                           constant_values=np.nan).reshape(-1, period), axis=0)
        return np.tile(seasonal_means, (int(np.ceil(ln / period)),))[:ln]

    def _calibrate_seasonality(self, data, period, trend_sign):
        """
        Calibrate seasonal data
        :param data: (numpy.ndarray) seasonal data of classical decomposition
        :param period: (int) period of seasonality
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :return: (numpy.ndarray) calibrated seasonal data
        """
        if trend_sign == 'div':
            return data / np.sum(data) * period
        elif trend_sign == 'sub':
            return data - np.sum(data) / len(data)

    def _plot_moving_averages(self, scale, data, name, period, units='units', axes=None):
        """
        Plot Moving Averages data of the source data
        :param data: (numpy.ndarray) source data
        :param name: (str) name for the title
        :param period: (int) period of moving averages
        :param units: (str) units of the variable
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Moving Averages Plot'))
            axes = figure.subplots()
        self._limit_xticks(scale, axes, period=period, grid=True)
        axes.plot(data)
        axes.set_title(f'{name}')
        axes.set_xlabel('time')
        axes.set_ylabel(units)

    def plot_moving_averages(self, scale, data, name, period, units='units', axes=None):
        """
        Creating moving averages data and supplies _plot_moving_averages function to create the plot
        :param scale: (numpy.ndarray) scale of data
        :param data: (numpy.ndarray) source data
        :param name: (str) name for the title
        :param period: (int) period of moving averages
        :param units: (str) units of the variable
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        ln = len(data)
        data_ = self._moving_averages(data, period)
        scale_ = self._moving_average_data_offset(ln, period)(scale)
        return self._plot_moving_averages(scale_, data_, name, period, units=units, axes=axes)

    def plot_seasonality(self, scale, data, name, period, trend_sign='div', number_of_periods=1, units='units', axes=None):
        """
        Calculates seasonality data using classical decomposition and supplies _plot_seasonality function to create the plot
        :param scale: (numpy.ndarray) scale of data
        :param data: (numpy.ndarray) source data
        :param name: (str) name for the title
        :param period: (int) seasonal period
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :param number_of_periods: (int) number of periods to plot (None to plot all the data)
        :param units: (str) units of the variable
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        _, rest_data = self._get_trend_rest_data(data, period, trend_sign=trend_sign)
        seasons = self._calibrate_seasonality(self._get_seasonality(rest_data, period), period, trend_sign)
        if number_of_periods:
            seasons = seasons[:number_of_periods * period]
            scale_ = scale[:number_of_periods * period]

        return self._plot_seasonality(scale_, seasons, name, period, units=units, axes=axes)

    def _plot_seasonality(self, scale, data, name, period, units='units', axes=None):
        """
        Plot seasonality using classical decomposition on source data
        :param scale: (numpy.ndarray) scale of data
        :param data: (numpy.ndarray) source data
        :param name: (str) name for the title
        :param period: (int) seasonal period
        :param units: (str) units of the variable
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Moving Averages Plot'))
            axes = figure.subplots()
        self._limit_xticks(scale, axes, period=period, grid=True)
        axes.plot(data)
        axes.set_title(f'{name}')
        axes.set_xlabel('time')
        axes.set_ylabel(units)
        return axes

    def _get_trend_seasonality_residuals(self, data, period, trend_sign, seasonal_sign):
        """
        Calculates trend, seasonal and residuals data after the application of classical decomposition
        :param data: (numpy.ndarray) source data
        :param period: (int) seasonal period
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :param seasonal_sign: (str) 'div' for multiplicative seasonality, 'sub' for additive seasonality
        :return: (tuple(numpy.ndarray)) trend, seasonal and residuals data
        """
        trend, rest_data = self._get_trend_rest_data(data, period, trend_sign)
        seasons = self._calibrate_seasonality(self._get_seasonality(rest_data, period), period, trend_sign)
        if seasonal_sign == 'sub':
            resids = rest_data - seasons
        elif seasonal_sign == 'div':
            resids = rest_data / seasons
        return trend, seasons, resids

    def plot_classical_decomposition(self, data, scale, name, period, number_of_periods=None, units='units',
                                     trend_sign='div', seasonal_sign='div'):
        """
        Plot data, trend, seasonality and residuals using classical decomposition method
        :param data: (numpy.ndarray) source data to applicate classical decomposition
        :param scale: (numpy.ndarray) Scale data of the variable to plot
        :param name: (str) name for the title
        :param period: (int) seasonal period
        :param number_of_periods: (int) number of periods to plot (None to plot all the data)
        :param units: (str) units of the variable
        :param trend_sign: (str) 'div' for multiplicative trend, 'sub' for additive trend
        :param seasonal_sign: (str) 'div' for multiplicative seasonality, 'sub' for additive seasonality
        :return: (None)
        """
        trend, season, resids = self._get_trend_seasonality_residuals(data, period, trend_sign=trend_sign,
                                                                      seasonal_sign=seasonal_sign)
        plt.interactive(True)
        figure = plt.figure(self._get_next_figure_name('Classical decomposition'))
        axes = figure.subplots(4, 1)

        slice_ = lambda x: x[:period * number_of_periods] if number_of_periods else x

        data_ = slice_(data)
        scale_ = slice_(scale)

        self._limit_xticks(scale_, axes[0], number_of_ticks=10, rotation=5, text=True, grid=True)
        self.plot(scale_, data_, '', units, axes=axes[0])
        plt.suptitle(f'{name} - data / trend / seasonality / residuals - period {period}')

        scale__ = slice_(self._moving_average_data_offset(len(scale), period)(scale))
        trend_ = slice_(trend)
        season_ = slice_(season)
        resids_ = slice_(resids)

        self._limit_xticks(scale__, axes[1], number_of_ticks=10, rotation=5, text=True, grid=True)
        self.plot(scale__, trend_, '', units, axes=axes[1])

        self._limit_xticks(scale__, axes[2], number_of_ticks=10, rotation=5, text=True, grid=True)
        self.plot(scale__, season_, '', units, axes=axes[2])

        self.plot_residuals(scale__, resids_ / data_, '', units, axes=axes[3])
        self._limit_xticks(scale__, axes[3], number_of_ticks=10, rotation=5, text=True, grid=True)


class VisualizeResults(Visualizer):

    def plot_forecasts(self, scale, actual, forecast, intervals=[], name='forecasts', units='units', axes=None):
        """
        Plots forecasts and confidential intervals
        :param scale: (numpy.ndarray) Scale data of the variable to plot
        :param actual: (numpy.ndarray) Actual values of the variable to plot
        :param forecast: (numpy.ndarray) Forecasting values of the variable to plot
        :param intervals: (numpy.ndarray) Prediction intervals of the forecasts
        :param name: (str) name for the title
        :param units: (str) units of the variable
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        intervals = [*zip(*intervals)]
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Forecast plot'))
            axes = figure.subplots()
        axes.plot(scale, actual)
        axes.plot(scale, forecast)
        if intervals:
            axes.fill_between(scale, intervals[0], intervals[1], color='orange', alpha=.2)
        axes.set_title(f'{name}')
        axes.set_xlabel('time')
        axes.set_ylabel(units)
        axes.legend(labels=['actuals', 'forecasts'])
        plt.xticks(scale if len(scale) <= 20 else scale[np.round(np.linspace(0, len(scale) - 1, 20),
                                                                 0).astype(np.int64).tolist()], rotation=30)

        return axes

    def plot_loss_by_epoch(self, loss_history, name='training loss', units='loss', axes=None):
        """
        Plots training loss by epochs
        :param loss_history: (list) history of training losses
        :param name: (str) name for the title
        :param units: (str) y-axes label units
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        self.plot(range(1, len(loss_history) + 1), loss_history, xlab='epochs', name=name, units=units, axes=axes)

    def plot_validation_by_epoch(self, validation_history, name='validation loss', units='loss', axes=None):
        """
        Plots training validation loss by epochs
        :param validation_history:
        :param name: (str) name for the title
        :param units: (str) y-axes label units
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        self.plot(range(1, len(validation_history) + 1), validation_history, xlab='epochs', name=name, units=units,
                  axes=axes)

    def plot_loss_by_time(self, epoch_times, loss_history, name='training loss', units='loss', axes=None):
        """
        Plots training loss by time
        :param loss_history:
        :param name: name for the title
        :param units: (str) y-axes label units
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        self.plot(np.cumsum(epoch_times), loss_history, name=name, units=units, axes=axes)

    def plot_validation_by_time(self, epoch_times, validation_history, name='validation loss', units='loss', axes=None):
        """
        Plots training validation loss by time
        :param validation_history:
        :param name: name for the title
        :param units: (str) y-axes label units
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) axes of the plot
        """
        self.plot(np.cumsum(epoch_times), validation_history, name=name, units=units, axes=axes)

    def plot_compare_models_loss(self, losses, names, times=None, units='loss', axes=None):
        """
        Compares loss or validation of different models by epoch or by time
        :param losses: (list(list)) loss or validation training values lists for different models
        :param names: (list(str)) names for legend labels
        :param units: (str) y-axes label units
        :param axes: (pyplot.axes) axes where the plot will be drawn. Set None to use a new figure.
        :return:  (pyplot.axes) axes of the plot
        """
        xlab = 'time'
        if isinstance(times, type(None)):
            xlab = 'epochs'
            times = [range(1, len(loss) + 1) for loss in losses]
        for loss, time in zip(losses, times):
            axes = self.plot(time, loss, xlab=xlab, units=units, axes=axes)
        axes.legend(labels=names)

        return axes
