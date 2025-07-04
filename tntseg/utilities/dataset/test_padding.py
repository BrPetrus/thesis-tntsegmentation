from split_data import add_padding
import numpy as np


def test_simple():
    data = np.random.random((10, 10, 10))
    res = add_padding(data, 0, 10, 0, 10, 0, 10, (10, 10, 10))
    assert res.shape == (10, 10, 10)
    assert np.all(res == data)

def test_expand_no_zero_padding_odd() -> None:
    data = np.random.random((10, 10, 10))
    res = add_padding(data, 1, 3, 1, 3, 1, 3, (5, 5, 5))
    assert res.shape == (5, 5, 5)
    assert np.all(data[0:5, 0:5, 0:5] == res)

def test_expand_no_zero_padding_even() -> None:
    data = np.random.random((10, 10, 10))
    res = add_padding(data, 1, 3, 1, 3, 1, 3, (6, 6, 6))
    assert res.shape == (6, 6, 6)
    assert np.all(data[0:6, 0:6, 0:6] == res)

def test_expand_zero_padding_even() -> None:
    data = np.random.random((10, 10, 10))
    res = add_padding(data, 0, 10, 0, 10, 0, 10, (20, 20, 20))
    assert res.shape == (20, 20, 20)
    assert np.all(data == res[5:-5, 5:-5, 5:-5])  # TODO: does this test that the expantion also copied surrounding data?

def test_expand_zero_padding_odd() -> None:
    data = np.random.random((10, 10, 10))
    res = add_padding(data, 0, 10, 0, 10, 0, 10, (21, 21, 21))
    assert res.shape == (21, 21, 21)
    assert np.all(data == res[5:-6, 5:-6, 5:-6])