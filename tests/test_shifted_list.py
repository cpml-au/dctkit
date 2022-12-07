from dctkit.math import shifted_list as sl


def test_shifted_list():
    boundary = sl.ShiftedList([0,1],-1)
    assert boundary[1] == 0

if __name__ == '__main__':
    test_shifted_list()