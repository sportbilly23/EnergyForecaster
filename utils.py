import datetime
import numpy as np
import pytz


def timedelta_to_str(seconds):
    """
    Convert seconds to string of time counter
    :param seconds: (int) seconds of a timer
    :return: (str) string with days, hours, minutes and seconds
    """
    t_delta = datetime.timedelta(seconds=seconds)
    days = t_delta.days
    hours = int(t_delta.seconds / 3600)
    mins = int((t_delta.seconds - hours * 3600) / 60)
    secs = int(t_delta.seconds % 60)
    return f'{days}d {hours:02}:{mins:02}:{secs:02}'


def calculate_to_date(to_date):
    """
    Giving a date in a tuple of integers, it returns the maximum computer date representation for this date
    ex. For the date (2012, 3, 1), it returns (2012, 3, 1, 23, 59, 59, 999999)
    :param to_date: (tuple(int)) the date
    :return: (tuple(int)) maximum computer date representation for the source date
    """
    to_date = list(to_date)
    for i in reversed(range(len(to_date))):
        if to_date[i] == 0:
            to_date.pop(i)
        else:
            break

    ln = len(to_date)
    if ln < 3:
        to_date = tuple(1 if i >= ln else to_date[i] + (1 if i == ln - 1 else 0) for i in range(3))
        if to_date[1] == 13:
            to_date = (to_date[0] + 1, 1, to_date[2])
        to_date = datetime.datetime(*to_date) - datetime.timedelta(microseconds=1)
    else:
        to_date = datetime.datetime(*to_date) + datetime.timedelta(days=(ln == 3), hours=(ln == 4), minutes=(ln == 5),
                                                                   seconds=(ln == 6), microseconds=(ln == 7))
    to_date = to_date - datetime.timedelta(microseconds=1)
    return tuple(list(datetime.datetime.timetuple(to_date)[:-3]) + [to_date.microsecond])


def get_tzinfo(tzone):
    """
    Returns tzinfo from a timezone
    :param tzone: (str) Timezone label (eg. 'Europe/Athens')
    :return: (DstTzInfo) Timezone Info
    """
    return datetime.datetime.now(pytz.timezone(tzone)).tzinfo


def arrays_are_equal(a, b):
    """
    Returns True or False if numpy arrays are equal or not
    :param a: (numpy.ndarray) the first numpy array
    :param b: (numpy.ndarray) the second numpy array
    :return: (bool) True or False if numpy arrays are equal or not
    """
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def timestamp_to_date_str(dates, timezone):
    """
    Returns date strings from timestamp data with respect of a timezone
    :param dates: (numpy.ndarray) timestamp data
    :param timezone: (pytz.timezone) timezone
    :return: (numpy.ndarray) string dates
    """
    return np.array([datetime.datetime.fromtimestamp(i, tz=timezone).strftime('%d/%m/%y %H:%M:%S') for i in dates])

