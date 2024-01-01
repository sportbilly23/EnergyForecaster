class Preprocessor:
    """
    Transforms/Normalizes/Extends datasets for Energy Forecaster
    """
    def __init__(self, ef):
        self._EF = ef
        self.datasets = None
