import numpy as np


class DTable:
    """
    Table that holds the data of an EF file and outputs graphs and statistics.
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.columns = self.data.dtype.names

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    # TODO: Θα μπούνε οι συναρτήσεις που δημιουργούν γραφήματα ή στατιστικά για δεδομένα ενός πίνακα