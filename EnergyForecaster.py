from DataController import DataController
from Preprocessor import Preprocessor
from ProcessController import ProcessController
from Statistics import Statistics
from Visualizer import Visualizer
import logging


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
        self.preprocessor = Preprocessor(self)
        self.process_controller = ProcessController(self)
        self.statistics = Statistics(self)
        self.visualizer = Visualizer(self)
        self.data_controller = DataController(self, folderpath)


if __name__ == '__main__':
    import time
    ef = EnergyForecaster('e:/test')
    # print(time.time())
    # ef.data_controller.import_csv(f'C:\\Users\\sportbilly\\Downloads\\weather.csv', skip=2)
    ef.data_controller.get_dataset('weather')
    # print(ef.data_controller.current_datasets.keys())
    # ef.data_controller.current_datasets['consumption']['ES_load_actual_entsoe_transparency'][5] = 47368
    # ef.data_controller.set_dataset('consumption', 'consumption2')
    # print(ef.data_controller.dataset_is_changed('consumption'))
    # try:
    #     ef.data_controller.close_dataset('consumption')
    # except KeyError as e:
    #     print(e)
    print(time.time())
