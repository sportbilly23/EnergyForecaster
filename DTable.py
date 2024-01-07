import numpy as np
from Preprocessor import Preprocessor


class DTable:
    """
    Table that holds the data of an EF file and outputs graphs and statistics.
    """
    def __init__(self, data, _EF):
        self.data = np.array(data)
        self.columns = self.data.dtype.names
        self.preprocessor = _EF.preprocessor

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    # TODO: Θα μπούνε οι συναρτήσεις που δημιουργούν γραφήματα, στατιστικά και μετατροπές για δεδομένα ενός πίνακα
    def log(self, base, column):
        return self.preprocessor.log(base, self.data[column])

    def log2(self, column):
        return self.preprocessor.log2(self.data[column])

    def log10(self, column):
        return self.preprocessor.log10(self.data[column])

    def ln(self, column):
        return self.preprocessor.ln(self.data[column])

    def exp(self, column):
        return self.preprocessor.exp(self.data[column])

    def exp2(self, column):
        return self.preprocessor.exp2(self.data[column])

    def boxcox(self, column, lamda):
        return self.preprocessor.boxcox(self.data[column], lamda)

    def minmax(self, column):
        return self.preprocessor.minmax(self.data[column])

    def standard(self, column, centered=True, devarianced=True):
        return self.preprocessor.standard(self.data[column], centered=centered, devarianced=devarianced)


