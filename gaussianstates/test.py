from utils.binary_string_utils import int_to_binary_array, read_binary_array, strings_with_weight
from scipy.spatial.distance import hamming
import random

n = 6
k = int(2 ** (n - 1) - n * (n - 1) / 2 - 1)

odd_weight = [
    item for sublist in
    [strings_with_weight(n, k) for k in range(1, n + 1, 2)]
    for item in sublist
]

max_distances = []

ints_bin = [
    [0,0,1,1,1,0],[0,1,0,1,1,0],[0,1,1,0,1,0],[0,1,1,1,0,0],
    [1,0,0,1,1,0],[1,0,1,0,1,0],[1,0,1,1,0,0],[1,1,0,0,1,0],
    [1,1,1,0,0,0],[1,1,0,1,1,1],[1,1,1,1,1,0],[0,1,1,1,0,0],
    [1,0,1,1,0,0],[1,1,0,1,0,0],[1,1,1,0,0,0],[1,1,1,0,1,1]
]

distances = []

for e in odd_weight:
    count = 0
    for arr in ints_bin:
        if hamming(e, arr) * n > 2:
            count += 1
    distances.append(count)
d_max = max(distances)
