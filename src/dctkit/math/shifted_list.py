from typing import Iterable


class ShiftedList(list):
    """List with indices shifted by an offset.

        Args:
            iterable: iterable used to initialize the list.
            off (int): offset.
    """

    def __init__(self, iterable: Iterable, off: int):
        super().__init__(iterable)
        self.off = off

    def __getitem__(self, key):
        return super().__getitem__(key + self.off)

    def __setitem__(self, key, value):
        return super().__setitem__(key + self.off, value)
