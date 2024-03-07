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


def timestamp_to_date_str(dates, timezone):
    return np.array([datetime.datetime.fromtimestamp(i, tz=timezone).strftime('%d/%m/%y %H:%M:%S') for i in dates])


def random_chunk(data, size=1):
    """
    Επιστρέφει κομμάτι μεγέθους size συνεχόμενων τιμών από μια χρονοσειρά
    :param data: (np.array ή list ή tuple) Χρονοσειρά
    :param size: (int) Μέγεθος συνεχόμενων τιμών
    :return: np.array ή list ή tuple
    """
    start = np.random.randint(1, len(data) - size + 1)
    return data[start: start + size]


def shuffle_chunks(data, size=1, num=1):
    """
    Επιστρέφει ανακατεμένη την λίστα των δεδομένων κόβοντάς την σε κομμάτια μεγέθους size. Η λίστα επιστρέφεται
    num φορές.
    :param data:(np.array ή list ή tuple) Χρονοσειρά
    :param size: μέγεθος κομματιών που θα κοπεί η λίστα
    :param num: Αριθμός επιστροφών τυχαία ανακατεμένων λιστών
    :return: np.array
    """
    ln = len(data)
    lsts = []
    for i in range(num):
        lst = []
        while len(lst) < ln:
            lst += random_chunk(data, size).tolist()
        if num < 2:
            return np.array(lst[: ln])
        lsts.append(lst[:ln])
    return np.array(lsts)