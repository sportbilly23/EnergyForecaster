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

    ef = EnergyForecaster('e:/test_4')
    ef.data_controller.get_dataset('weather')
    ef.data_controller.get_dataset('consumption')
    ef.process_controller.get_process('process_1')
    # ef.process_controller.set_voting_model('vote_2', ['mlp_100', 'random_forest', 'conv1d'], add_to_process=True)
    # ef.process_controller.update_process()
    ef.process_controller.process.plot_forecasts('vote_2', data_part='test', start=4012, steps=24, alpha=.05,
                                                 intervals_from_validation=True)
    print(time.time())
