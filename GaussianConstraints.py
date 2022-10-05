import copy

from GaussianStates import *
import numpy as np
from scipy.special import comb
import math


def kbits(n, k) -> list[np.ndarray]:
    bit_strings = []
    limit=1<<n
    val=(1<<k)-1
    while val<limit:
        bit_strings.append(np.array([*"{0:0{1}b}".format(val,n)], dtype=int))
        minbit=val&-val #rightmost 1 bit
        fillbit = (val+minbit)&~val  #rightmost 0 to the left of that bit
        val = val+minbit | (fillbit//(minbit<<1))-1
    if k == n / 2:
        bit_strings = bit_strings[0:int(len(bit_strings)/2)]# will break for odd n
    return bit_strings


def read_binary_array(bin_arr: np.ndarray):
    res = 0
    l = bin_arr.size
    for i in range(l):
        res += 2 ** (l - i - 1) * bin_arr[i]
    return int(res)


dim = 10
gaussian = nn_gaussian_states(1, dim, 0)

print('Starting')
constraints = []
for k in range(1, 2 * math.floor((dim + 2) / 4), 2):
    targets = kbits(dim, k)
    print(k,targets)
    for target in targets:
        constraint = np.zeros((dim, 2))
        for i in range(dim):
            x = copy.deepcopy(target)
            x[i] = (x[i] + 1) % 2
            y = (x + 1) % 2
            x_int = read_binary_array(x)
            y_int = read_binary_array(y)
            constraint[i, :] = np.array([x_int, y_int], dtype=int)
        constraints.append(constraint)

for c in constraints:
    print(c)
print(len(constraints))
