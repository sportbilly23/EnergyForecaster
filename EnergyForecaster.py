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
    ef = EnergyForecaster('e:/test_3')
    # print(time.time())
    # ef.data_controller.import_csv(f'C:\\Users\\sportbilly\\Downloads\\weather.csv', skip=2)
    # ef.data_controller.import_csv(f'C:\\Users\\sportbilly\\Downloads\\weather.csv', h5_name='sb', skip=2)
    # ef.data_controller.import_csv(f'C:\\Users\\sportbilly\\Downloads\\consumption.csv', h5_name='sb2')
    # ef.data_controller.get_dataset('sb2')
    # ef.data_controller.get_dataset('weather')
    # ef.data_controller.get_dataset('sb')
    # sb = ef.data_controller.datasets['sb']
    # sb2 = ef.data_controller.datasets['sb2']
    # ef.data_controller.import_csv('C:\\Users\\sportbilly\\Downloads\\consumption.csv', h5_name='test')
    # sb.to_timestamp('time', '%Y-%m-%d %H:%M:%S', assign='inplace', tzone='utc')
    # sb.minmax(column='temperature', assign='inplace')
    # sb.set_units('temperature', 'oC')
    # sb.weekday(column='time', assign='add')
    # ef.data_controller.update_dataset('sb')

    # sb.to_timestamp('time', '%Y-%m-%d %H:%M:%S', tzone='utc', assign='inplace')
    # sb.weekday('time', 'one-hot', assign='add')
    # sb.attach_scale('temperature', 'time')
    # sb.attach_scale('precipitation', 'time')
    # sb.attach_scale('irradiance_surface', 'time')
    # sb.attach_scale('snow_mass', 'time')
    # sb.attach_scale('irradiance_toa', 'time')
    # sb.attach_scale('cloud_cover', 'time')
    # sb.attach_scale('air_density', 'time')
    # sb.attach_scale('snowfall', 'time')

    # sb.minmax('temperature', assign='inplace')
    # sb.minmax('precipitation', assign='inplace')
    # sb.minmax('irradiance_surface', assign='inplace')
    # sb.minmax('snow_mass', assign='inplace')
    # sb.minmax('irradiance_toa', assign='inplace')
    # sb.minmax('cloud_cover', assign='inplace')
    # sb.minmax('air_density', assign='inplace')
    # sb.minmax('snowfall', assign='inplace')
    # sb2.to_timestamp('cet_cest_timestamp', '%Y-%m-%dT%H:%M:%S%z', tzone='Europe/Madrid', assign='inplace')
    # sb2.make_scale('cet_cest_timestamp')
    # sb2.attach_scale('ES_load_actual_entsoe_transparency', 'cet_cest_timestamp')
    # sb2.make_target('ES_load_actual_entsoe_transparency')
    # ef.data_controller.update_dataset('sb2')
    # ef.data_controller.update_dataset('sb')

    # t = sb.downgrade_data_frequency('temperature', 'week')
    # sb.plot('temperature')
    # sb.plot('temperature', freq='day')
    # sb.plot('temperature', freq='week')
    # import numpy as np
    # sb.plot('temperature', freq='month', func=np.mean, from_date=(2000, 1, 1), to_date=(2000, 12, 31))
    # sb.plot_seasons('temperature', 'weekly', from_date=(2000, 1, 3), to_date=(2000, 1, 30), func=np.mean, freq='hour')
    # sb.plot_schemas(['precipitation', 'snowfall', 'air_density', 'temperature'], from_date=(2000, 1, 1), to_date=(2000, 12, 31), freq='week')
    # sb.hist('temperature', plot_norm=True, density=True)
    # sb.detach_scale('precipitation')
    # sb.attributes['precipitation']
    # ef.data_controller.close_dataset('sb')
    # sb.plot_acf('temperature', diffs=(24, 1, 1), nlags=25)
    # sb.plot_pacf('temperature', diffs=(24, 1, 1), nlags=25)
    # sb.lagged_series('weekday', name='weekday', lags=(0, 2, 3, 4, 5, 6, 7), assign='inplace')
    # sb.lagged_series('temperature', 'temp', lags=(1, 2), assign='add')
    # ef.process_controller.set_process('sb', lags=3, black_lags=2, target_length=24)
    # ef.process_controller.get_process('process_2')
    # ef.process_controller.insert_data('sb', ['temperature', 'cloud_cover',
    #                                          'irradiance_surface', 'precipitation'], no_lags=False)
    # ef.process_controller.insert_data('sb', ['time_204588272', 'time_588562846', 'time_728017024', 'time_675637917'])

    # ef.process_controller.insert_data('sb2', ['ES_load_actual_entsoe_transparency'])
    # ef.process_controller.update_process()

    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(1, 1, 1)),
    #                                 'arima_111', 'statsmodels')
    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(0, 0, 0)),
    #                                 'arima_000_2', 'statsmodels')
    # ef.process_controller.set_model(RandomForestRegressor(), 'random_forest', 'sklearn')
    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(3, 1, 0)),
    #                                 'arima_310', 'statsmodels')
    # ef.process_controller.update_process()
    # ef.process_controller.fit_models()
    # aic = ef.process_controller.process.aic()

    # ef.process_controller.insert_data('sb', ['temperature', 'cloud_cover',
    #                                          'irradiance_surface', 'precipitation'], no_lags=False)
    # ef.process_controller.insert_data('sb', ['time_204588272', 'time_588562846', 'time_728017024', 'time_675637917'])
    #
    # ef.process_controller.insert_data('sb2', ['ES_load_actual_entsoe_transparency'])
    # ef.process_controller.update_process()
    # ef.process_controller.remove_process('sb2')
    # ef.process_controller.process.plot_forecast('arima_000', 'validation', steps=200, start=200,
    #                                             intervals_from_residuals=True, alpha=0.55)
    # axes = sb2.plot_seasonality('ES_load_actual_entsoe_transparency', 7 * 24, number_of_periods=3, trend_sign='div')
    # axes = sb2.plot_seasonality('ES_load_actual_entsoe_transparency', 7 * 24, number_of_periods=3, trend_sign='sub')
    # sb2.plot_classical_decomposition('ES_load_actual_entsoe_transparency', 168, number_of_periods=3, trend_sign='sub', seasonal_sign='sub')
    # sb2.plot_seasonality('ES_load_actual_entsoe_transparency', period = 7 * 24, number_of_periods=4)
    # ef.process_controller.close_process()
    # ef.data_controller.import_csv('F:/My Drive/data/apartment-weather/apartment2014.csv')
    # ef.process_controller.get_process('process_3')
    # ef.process_controller.insert_data('sb', ['temperature', 'irradiance_surface', 'cloud_cover', 'precipitation'],
    #                                   no_lags=False)
    # print(sb.data_summary())
    # ef.process_controller.remove_process('process_3')
    # ef.process_controller.run_process_script('c:/users/sportbilly/Downloads/script')
    # ef.process_controller.fit_models()

    # ef.data_controller.get_dataset('weather')
    # ef.data_controller.get_dataset('consumption')
    # ef.process_controller.set_process('process_3', lags=3, black_lags=1)
    # ef.process_controller.get_process('process_3')
    # ef.data_controller.import_csv(f'C:\\Users\\sportbilly\\Downloads\\weather.csv', skip=2)
    # ef.data_controller.import_csv('C:\\Users\\sportbilly\\Downloads\\consumption.csv')
    # ef.process_controller.run_process_script('c:/users/sportbilly/Downloads/script')
    # ef.process_controller.run_process_script('c:/users/sportbilly/Downloads/script_2')

    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(0, 0, 0)),
    #                                 'arima_000')
    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(0, 0, 0),
    #                                         seasonal_order=(0, 1, 0, 24)),
    #                                 'arima_000_s010_24')
    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(0, 0, 0),
    #                                         seasonal_order=(0, 0, 0, 24)),
    #                                 'arima_000_s000_24')
    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(0, 0, 0),
    #                                         seasonal_order=(1, 0, 0, 24)),
    #                                 'arima_000_s100_24')
    # ef.process_controller.set_model(SARIMAX(exog=ef.process_controller.process.get_data(),
    #                                         endog=ef.process_controller.process.get_target(), order=(0, 0, 0),
    #                                         seasonal_order=(0, 0, 1, 24)),
    #                                 'arima_000_s001_24')
    # ef.process_controller.set_model(RandomForestRegressor(), 'random_forest')
    # ef.process_controller.set_model(TransformerModel(24, 24, n_epochs=30), 'transformer')
    # ef.process_controller.update_process()
    # ef.process_controller.fit_models()
    # print(ef.process_controller.process.mape('arima_000_2', 'validation'))
    # print(ef.process_controller.process.mape('random_forest', 'validation'))
    # ef.process_controller.process.plot_forecast('transformer', 'validation')
    # print(ef.process_controller.process.evaluation_summary('validation'))
    # ef.process_controller.process.plot_forecast('arima_000', 'validation', steps=240, intervals_from_residuals=True, alpha=0.05)
    # ef.process_controller.process.plot_forecast('random_forest', 'validation', steps=240,
    #                                             intervals_from_residuals=False, alpha=0.05)
    # ef.data_controller.get_dataset('consumption')
    # ef.data_controller.datasets['consumption'].plot_seasons('ES_load_actual_entsoe_transparency', period='daily',
    #                                                         from_date=(2015, 3, 1), to_date=(2015, 3, 7), freq='hour')
    # ef.process_controller.process.plot_shapes(['temperature_lag-1', 'ES_load_actual_entsoe_transparency'],
    #                                           data_part='validation')
    # ef.process_controller.process.extend_fit('mlp_100', 2000)

    # model = TorchModel(54, 'cuda')
    # model.add_components(
    #     [
    #         'linear', {'out_features': 100},
    #         'relu', {},
    #         'linear', {'out_features': 1},
    #         'adam', {'lr': 0.005},
    #         'mse', {}
    #     ]
    # )
    # ef.process_controller.set_model(model, 'linear_100r', fit_params={'batch_size': 64}, add_to_process=True)
    # model.train(ef.process_controller.process.get_data(), ef.process_controller.process.get_target(), 300, 128,
    #             validation_data=ef.process_controller.process.get_data('validation'),
    #             validation_target=ef.process_controller.process.get_target('validation'),
    #             func=ef.results_statistics.mape)
    # del model

    # ef.process_controller.process.plot_forecast('linear_100r', 'validation', alpha=.05, steps=300,
    #                                             intervals_from_residuals=True)
    # ef.process_controller.process.extend_fit('linear_100r', n_epochs=100)

    # ef.process_controller.set_process('sb', 72, 24, 24)
    ef.process_controller.get_process('sb')
    # m = TorchModel(330, 'cuda', components=
    # [
    #     'gru', {'input_size': 330, 'hidden_size': 512, 'num_layers': 2, 'dropout': 0.1},
    #     'linear', {'out_features': 1},
    #     'adam', {'lr': 0.005},
    #     'mse', {}
    # ]
    #                )
    # m.train(ef.process_controller.process.get_data(), ef.process_controller.process.get_target(), 1000, 128,
    #             validation_data=ef.process_controller.process.get_data('validation'),
    #             validation_target=ef.process_controller.process.get_target('validation'),
    #             func=ef.results_statistics.mape)
    # ef.process_controller.set_model(
    #     name='lstm',
    #     fit_params={'batch_size': 16, 'func': ef.results_statistics.mape},
    #     model=TorchModel(
    #         input_size=330,
    #         device='cuda',
    #         components=[
    #             'rnn', {'input_size': 330, 'hidden_size': 512, 'num_layers': 1},
    #             'linear', {'out_features': 64},
    #             'linear', {'out_features': 1},
    #             'adam', {'lr': 0.005},
    #             'mse', {}
    #         ]
    #     )
    # )
    # ef.process_controller.set_model(
    #     name='transformer',
    #     fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
    #     model=TorchModel(
    #         input_size=330,
    #         device='cuda',
    #         components=[
    #             'transformer', {'d_model': 330, 'nhead': 10},
    #             'linear', {'out_features': 64},
    #             'linear', {'out_features': 1},
    #             'adam', {'lr': 0.005},
    #             'mse', {}
    #         ]
    #     )
    # )
    # ef.process_controller.set_model(
    #     name='linear',
    #     fit_params={'batch_size': 64, 'func': ef.results_statistics.mape},
    #     model=TorchModel(
    #         input_size=330,
    #         device='cuda',
    #         components=[
    #             'linear', {'out_features': 1024},
    #             'mish', {},
    #             'linear', {'out_features': 1024},
    #             'relu', {},
    #             'linear', {'out_features': 1},
    #             'adam', {'lr': 0.005},
    #             'l1', {}
    #         ]
    #     )
    # )
    # ef.process_controller.process.fit_model('linear', n_epochs=300, use_torch_validation=True)
    # ef.process_controller.process.plot_forecast('gru', alpha=.05, intervals_from_validation=False)
    # ef.process_controller.process.fit_model('gru', n_epochs=300, use_torch_validation=True)
    # ef.process_controller.set_voting_model('vote_1', ['linear', 'random_forest'], True)
    print(ef.process_controller.process.evaluation_summary(['vote_1'], 'validation'))
    print(time.time())
