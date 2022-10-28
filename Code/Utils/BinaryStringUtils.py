import numpy as np
from typing import List


def strings_with_weight(n, k) -> List[np.ndarray]:
    if k > n:
        raise Exception(f'Cannot get weight {k} strings of length {n}')
    bit_strings = []
    if k == 0:
        return bit_strings
    limit = 1 << n
    val = (1 << k) - 1
    while val < limit:
        bit_strings.append(np.array([*"{0:0{1}b}".format(val, n)], dtype=int))
        min_bit = val & -val  # rightmost 1 bit
        fillbit = (val + min_bit) & ~val  # rightmost 0 to the left of that bit
        val = val + min_bit | (fillbit // (min_bit << 1)) - 1
    return bit_strings


def read_binary_array(bin_arr: np.ndarray) -> int:
    res = 0
    length = len(bin_arr)
    for i in range(length):
        res += 2 ** (length - i - 1) * bin_arr[i]
    return int(res)


def int_to_binary_array(val: int, n: int) -> np.ndarray:
    return np.array([*"{0:0{1}b}".format(val, n)], dtype=int)
