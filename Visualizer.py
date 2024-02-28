import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Creates graphical visualizations for Energy Forecaster
    """
    def __init__(self, ef):
        self._EF = ef
        self.datasets = None

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

    def plot(self, scale, data, name='data', units='units', axes=None):
        """
        Creates a plot to visualize variable into time
        :param scale: (numpy.ndarray) Scale data of the variable to plot
        :param data: (numpy.ndarray) Data of the variable to plot
        :param name: (str) Names of the variable
        :param units: (str) Units of the variable
        :param axes: (pyplot.axes) Axes where the plot will be drawn. Set None to use a new figure.
        :return: (pyplot.axes) Axes of the plot
        """
        plt.interactive(True)
        if not axes:
            figure = plt.figure(self._get_next_figure_name('Plot'))
            axes = figure.subplots()
        axes.plot(scale, data)
        axes.set_title(f'{name}')
        axes.set_xlabel('time')
        axes.set_ylabel(units)
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
        title = ' vs '.join(names)
        for data, name in zip(datas, names):
            data = self._EF.preprocessor.minmax(data)[0]
            axes = self.plot(scale, data, title, '', axes)
            legend.append(name)

        axes.legend(labels=legend)

    def plot_seasons(self, datas, name, units, legend_lines, axes=None):
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

    # def _plot_calculate_data_scale_units(self, column, from_date, to_date, reverse_transform, freq, func):
    #     data, units = (self.reverse_trans(column), self.attributes[column]['units']) if reverse_transform else\
    #         (self.data[column], self.get_units(column))
    #     scale_column = self._get_scale(column)
    #     scale = self.data[scale_column]
    #     date_mask = self._preprocessor._date_mask(scale, from_date, to_date, self.attributes[scale_column]['timezone'])
    #     scale = np.array(scale)[date_mask]
    #     data = data[date_mask]
    #
    #     if self._compatible_data([column]) and self._no_scale_in_data([column]):
    #         if freq:
    #             data, scale = self.downgrade_data_frequency(column, freq, from_date, to_date, func, reverse_transform)
    #             if freq == 'week':
    #                 years = set([s[0] for s in scale])
    #                 maxs = [max([s[1] for s in scale if s[0] == y]) + 1 for y in years]
    #                 dct = dict(zip(years, maxs))
    #                 scale = [i[0] + i[1] / dct[i[0]] for i in scale]
    #             scale = np.array(scale)
    #         else:
    #             scale = self._preprocessor._timestamps_to_dates(scale, tz=self.attributes[scale_column]['timezone'])
    #     return data, scale, units


class VisualizeResults(Visualizer):

    pass
