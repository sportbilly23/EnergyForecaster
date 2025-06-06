from DataController import DataController
from Preprocessor import Preprocessor
from ProcessController import ProcessController
from Statistics import StatsData, StatsResults
from Visualizer import VisualizeData, VisualizeResults
from Models import *


class EnergyForecaster:
    """
    Framework for Energy forecasting

    Contains all necessary tools for:
        - Data Analysis/Preprocessing
        - Multi-Model Training and Evaluation
        - Results Analysis

    Author: Vasileios Konstas
    """
    def __init__(self, folderpath):
        """

        :param filename: (str) The name of EF's directory/main file
        """
        self.Model = Model
        self.preprocessor = Preprocessor(self)
        self.process_controller = ProcessController(self)
        self.data_statistics = StatsData(self)
        self.results_statistics = StatsResults(self)
        self.data_visualizer = VisualizeData(self)
        self.results_visualizer = VisualizeResults(self)
        self.data_controller = DataController(self, folderpath)


if __name__ == '__main__':
    import time

    def test_new_ef(ef_name, process_name):
        ef = EnergyForecaster(f'e:/{ef_name}')

        ef.data_controller.import_csv(f'C:\\Users\\sportbilly\\Downloads\\weather.csv', skip=2)
        ef.data_controller.import_csv('C:\\Users\\sportbilly\\Downloads\\consumption.csv')
        ef.process_controller.run_process_script('c:/users/sportbilly/Downloads/script')

        ef.process_controller.set_process(process_name, 72, 24, 24)
        ef.process_controller.get_process(process_name)
        ef.process_controller.run_process_script('c:/users/sportbilly/Downloads/script_2')

        ef.process_controller.set_model(ef.Model.SARIMAX(exog=ef.process_controller.process.get_data(),
                                                         endog=ef.process_controller.process.get_target(),
                                                         order=(0, 0, 0)), 'arima_000', add_to_process=True)
        ef.process_controller.set_model(ef.Model.RandomForestRegressor(), 'random_forest', add_to_process=True)
        ef.process_controller.set_model(ef.Model.MLPRegressor(), 'mlp_100', add_to_process=True)

        ef.process_controller.set_model(
            name='linear',
            fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
            add_to_process=True,
            model=ef.Model.TorchModel(
                input_size=330,
                device='cuda',
                components=[
                    'linear', {'out_features': 1024},
                    'mish', {},
                    'linear', {'out_features': 1024},
                    'relu', {},
                    'linear', {'out_features': 1},
                    'adam', {'lr': 0.005},
                    'mse', {}
                ]
            )
        )

        ef.process_controller.set_model(
            name='lstm',
            fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
            add_to_process=True,
            model=ef.Model.TorchModel(
                input_size=330,
                device='cuda',
                components=[
                    'lstm', {'input_size': 330, 'hidden_size': 512, 'num_layers': 1},
                    'linear', {'out_features': 64},
                    'linear', {'out_features': 1},
                    'adam', {'lr': 0.005},
                    'mse', {}
                ]
            )
        )

        ef.process_controller.set_model(
            name='conv1d',
            fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
            add_to_process=True,
            model=ef.Model.TorchModel(
                input_size=330,
                device='cuda',
                components=[
                    'conv1d', {'out_channels': 16, 'kernel_size': 15, 'stride': 7},
                    'mish', {},
                    'linear', {'out_features': 1024},
                    'relu', {},
                    'linear', {'out_features': 1},
                    'adam', {'lr': 0.005},
                    'mse', {}
                ]
            )
        )

        ef.process_controller.update_process()

        ef.process_controller.process.fit_models(n_epochs=300, use_torch_validation=True)

    def test_new_models(ef_name, process_name):
        ef = EnergyForecaster(f'e:\\{ef_name}')

        ef.process_controller.get_process(process_name)

        # ef.process_controller.process.remove_model('linear')
        # ef.process_controller.process.remove_model('conv1d')
        # ef.process_controller.process.remove_model('lstm')
        ef.process_controller.set_model(
            name='linear',
            fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
            add_to_process=True,
            model=ef.Model.TorchModel(
                input_size=330,
                device='cuda',
                components=[
                    'linear', {'out_features': 1024},
                    'mish', {},
                    'linear', {'out_features': 1024},
                    'relu', {},
                    'linear', {'out_features': 1},
                    'adam', {'lr': 0.005},
                    'mse', {}
                ]
            )
        )

        ef.process_controller.set_model(
            name='lstm',
            fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
            add_to_process=True,
            model=ef.Model.TorchModel(
                input_size=330,
                device='cuda',
                components=[
                    'lstm', {'input_size': 330, 'hidden_size': 512, 'num_layers': 1},
                    'linear', {'out_features': 64},
                    'linear', {'out_features': 1},
                    'adam', {'lr': 0.005},
                    'mse', {}
                ]
            )
        )

        ef.process_controller.set_model(
            name='conv1d',
            fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
            add_to_process=True,
            model=ef.Model.TorchModel(
                input_size=330,
                device='cuda',
                components=[
                    'conv1d', {'out_channels': 16, 'kernel_size': 15, 'stride': 7},
                    'mish', {},
                    'linear', {'out_features': 1024},
                    'relu', {},
                    'linear', {'out_features': 1},
                    'adam', {'lr': 0.005},
                    'mse', {}
                ]
            )
        )

        ef.process_controller.update_process()

        ef.process_controller.process.fit_models(n_epochs=300, use_torch_validation=True)

    def test_4_process_1():
        ef = EnergyForecaster('e:/test_4')
        ef.data_controller.get_dataset('weather')
        ef.data_controller.get_dataset('consumption')
        ef.process_controller.get_process('process_1')

    def example():
        ef = EnergyForecaster('e:/example')
        ef.data_controller.get_dataset('weather')
        ef.data_controller.get_dataset('consumption')
        weather = ef.data_controller.datasets['weather']
        consumption = ef.data_controller.datasets['consumption']
        # ef.process_controller.set_process('process_1', lags=72, black_lags=24, measure_period=24, update_file=True)
        ef.process_controller.get_process('process_1')
        # ef.process_controller.process.insert_data(dataset='weather',
        #                                           columns=['precipitation', 'temperature', 'irradiance_surface',
        #                                                    'cloud_cover'], no_lags=False)
        # ef.process_controller.process.insert_data(dataset='consumption', columns=['ES_load_actual_entsoe_transparency'],
        #                                           no_lags=True)
        # ef.process_controller.process.plot_forecasts('random_forest', alpha=.05, data_part='test', start=0, steps=500)
        # ef.process_controller.process.plot_compare_models_loss(names=['linear', 'lstm', 'conv1d'], time=True,
        #                                                        use_validation=False)
        import matplotlib.pyplot as plt
        # plt.interactive(True)
        # fig2 = plt.figure('histograms', figsize=(12, 10))
        # fig2.subplots(nrows=2, ncols=3)
        # plt.subplots_adjust(hspace=0.4)

        plt.interactive(True)
        fig3 = plt.figure('histograms', figsize=(12, 10))
        fig3.subplots(nrows=2, ncols=3)
        plt.subplots_adjust(hspace=0.5)

        for i, model in enumerate(ef.process_controller.process.models):
            ef.process_controller.process.plot_residuals(name=model, start=0, steps=1000, axes=fig3.axes[i],
                                                         torch_best_valid=False)

        dates = [(2015, 1, 1, 0, 0, 0), (2015, 2, 1, 10, 0, 0), (2015, 3, 14, 12, 0, 0), (2015, 3, 15, 12, 0, 0),
                 (2015, 9, 29, 23, 0, 0), (2015, 9, 30, 23, 0, 0)]
        import datetime, pytz

        for dt in dates:
            dat = datetime.datetime(*dt, tzinfo=pytz.utc)
            print(dat.ctime())
            print('\tweekend')
            print(f'\t\tone-hot: {ef.preprocessor.weekend(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print('\tweekday')
            print(f'\t\tone-hot: {ef.preprocessor.weekday(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tcos-sin: {ef.preprocessor.weekday(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tvar: {ef.preprocessor.weekday(data=[dat.timestamp()], time_zone=pytz.utc, mode="var")[0]}')
            print('\tmonthday')
            print(f'\t\tone-hot: {ef.preprocessor.monthday(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tcos-sin: {ef.preprocessor.monthday(data=[dat.timestamp()], time_zone=pytz.utc, mode="cos-sin")[0]}')
            print(f'\t\tvar: {ef.preprocessor.monthday(data=[dat.timestamp()], time_zone=pytz.utc, mode="var")[0]}')
            print('\tday_hour')
            print(f'\t\tone-hot: {ef.preprocessor.day_hour(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tcos-sin: {ef.preprocessor.day_hour(data=[dat.timestamp()], time_zone=pytz.utc, mode="cos-sin")[0]}')
            print(f'\t\tvar: {ef.preprocessor.day_hour(data=[dat.timestamp()], time_zone=pytz.utc, mode="var")[0]}')
            print('\tyear_day')
            print(f'\t\tone-hot: {ef.preprocessor.year_day(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tcos-sin: {ef.preprocessor.year_day(data=[dat.timestamp()], time_zone=pytz.utc, mode="cos-sin")[0]}')
            print(f'\t\tvar: {ef.preprocessor.year_day(data=[dat.timestamp()], time_zone=pytz.utc, mode="var")[0]}')
            print('\tyear_week')
            print(f'\t\tone-hot: {ef.preprocessor.year_week(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tcos-sin: {ef.preprocessor.year_week(data=[dat.timestamp()], time_zone=pytz.utc, mode="cos-sin")[0]}')
            print(f'\t\tvar: {ef.preprocessor.year_week(data=[dat.timestamp()], time_zone=pytz.utc, mode="var")[0]}')
            print('\tyear_month')
            print(f'\t\tone-hot: {ef.preprocessor.year_month(data=[dat.timestamp()], time_zone=pytz.utc)[0]}')
            print(f'\t\tcos-sin: {ef.preprocessor.year_month(data=[dat.timestamp()], time_zone=pytz.utc, mode="cos-sin")[0]}')
            print(f'\t\tvar: {ef.preprocessor.year_month(data=[dat.timestamp()], time_zone=pytz.utc, mode="var")[0]}')
            print()

        print(time.time())

    # example()

    def solar():
        ef = EnergyForecaster('e:/solar')
        # ef.data_controller.import_csv(filename='C:\\Users\\sportbilly\\Desktop\\solar\\ENTSOE_data.csv',
        #                               str_to_nan=('n/e', ''))
        # ef.data_controller.import_csv('C:\\Users\\sportbilly\\Desktop\\solar\\wind_energy_data.csv')
        # ef.data_controller.import_csv('C:\\Users\\sportbilly\\Desktop\\solar\\solar_energy_data.csv')
        # ef.data_controller.import_csv('C:\\Users\\sportbilly\\Desktop\\solar\\historic_weather_data.20170101-20210102.csv')
        return ef
    # ef = solar()

    def test_5():
        ef = EnergyForecaster('e:/example2')
        ef.data_controller.get_dataset('weather')
        ef.data_controller.get_dataset('consumption')
        return ef

    def test_6():
        ef = EnergyForecaster('e:/example3')
        ef.data_controller.import_csv(
            'E:/data2/Original Data/renewables_ninja_country_GR_air-density_merra-2_land-wtd.csv',
            h5_name='air-density', skip=4)
        return ef

    ef = EnergyForecaster('e:/example3')
    # ef.data_controller.import_csv(
    #     'E:/data2/Original Data/Total Load - Day Ahead _ Actual_201501010000-201601010000.csv',
    #     h5_name='total-load-2015', quotes='"', str_to_nan=['N/A', ''], all_float=True)
    # ef.data_controller.import_csv(
    #     'E:/data2/Original Data/Total Load - Day Ahead _ Actual_201601010000-201701010000.csv',
    #     h5_name='total-load-2016', quotes='"', str_to_nan=['N/A', ''], all_float=True)
    ef.data_controller.get_all_datasets()
    ef.data_controller.get_dataset_names()
    ef.process_controller.get_process('process_1')
    # ef.data_controller.datasets['precipitation'].average_data(['"EL41"', '"EL42"', '"EL43"'], 'avg', assign=None)
    print(time.time())
