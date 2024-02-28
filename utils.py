import datetime
import numpy as np


def calculate_to_date(to_date):
    ln = len(to_date)
    a = datetime.datetime(*to_date) + \
        datetime.timedelta(days=(ln == 3), hours=(ln == 4), minutes=(ln == 5), seconds=(ln == 6),
                           microseconds=(ln == 7)) - datetime.timedelta(microseconds=1)
    return tuple(list(datetime.datetime.timetuple(a)[:-3]) + [a.microsecond])


def arrays_are_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True