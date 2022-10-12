import numpy as np


def strings_with_weight(n, k) -> list[np.ndarray]:
    bit_strings = []
    limit = 1 << n
    val = (1 << k) - 1
    while val < limit:
        bit_strings.append(np.array([*"{0:0{1}b}".format(val, n)], dtype=int))
        minbit = val & -val  # rightmost 1 bit
        fillbit = (val + minbit) & ~val  # rightmost 0 to the left of that bit
        val = val + minbit | (fillbit // (minbit << 1)) - 1
    if (n % 2 == 0 and k == n / 2) or (n % 2 == 1 and k == (n - 1) / 2):
        bit_strings = bit_strings[0:int(len(bit_strings) / 2)]
    return bit_strings


def read_binary_array(bin_arr: np.ndarray) -> int:
    res = 0
    l = len(bin_arr)
    for i in range(l):
        res += 2 ** (l - i - 1) * bin_arr[i]
    return int(res)


def int_to_binary_array(val: int, n: int) -> np.ndarray:
    return np.array([*"{0:0{1}b}".format(val, n)], dtype=int)
