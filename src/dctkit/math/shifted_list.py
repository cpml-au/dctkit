class ShiftedList(list):
    """List with indices shifted by an offset. Subclass of list.

        Args:
            iterable: iterable used to initialize the list.
            off (int): offset.
    """

    def __init__(self, iterable, off):
        super().__init__(iterable)
        self.off = off

    def __getitem__(self, key):
        return super().__getitem__(key + self.off)

    def __setitem__(self, key, value):
        return super().__setitem__(key + self.off, value)
