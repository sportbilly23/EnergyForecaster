from collections import UserDict


class DictNoDupl(UserDict):
    """
    Dictionairy that do not accept changes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """
        Check if the new key is already in the dictionairy. Raises exception if this is True.
        :param key: (var) The new key
        :param value: (var) The new value
        :return: None
        """
        if key in self.keys():
            raise KeyError(f'Key {key} exists already. Choose another key or remove existing member')
        super().__setitem__(key, value)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def __iter__(self):
        return iter(self.data.keys())

