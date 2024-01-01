class ProcessController:
    """
    Manages processes for Energy Forecaster
    """

    def __init__(self, ef):
        self._EF = ef
        self.datasets = None
        self.processes = None
        self.current_process =  None
