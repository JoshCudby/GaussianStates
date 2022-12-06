import pickle
import numpy as np
import qutip
from gaussianstates.utils.binary_string_utils import read_binary_array, strings_with_weight

with open('data/GaussianRank/gaussian_rank_n_8', 'rb') as f:
    data = pickle.load(f)

data_x = data.x
x = data_x.reshape(3, 258)

complex_x = np.zeros((3, 128), dtype='complex')

for i in range(3):
    for i2 in range(129):
        complex_x[i][i2] = x[i][2 * i2] + 1j * x[i][2 * i2 + 1]

[np.sum(data.x[a: a + 256] ** 2) for a in [0, 258, 516]]
[np.sum(abs(complex_x[0, 0:128]) ** 2) for i in [0, 1, 2]]

y = sum([complex_x[i, 0:128] * complex_x[i, 128] for i in range(3)])

n = 8
even_weight_bin = [
    item for sublist in
    [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
    for item in sublist
]
even_weight_labels = [read_binary_array(e) for e in even_weight_bin]

state = qutip.basis(256, 0)
for index, entry in enumerate(y):
    state += qutip.basis(256, even_weight_labels[index]) * entry
state -= qutip.basis(256, 0)

magic_state = 0.5 * (qutip.basis(256, 0) + qutip.basis(256, 15) + qutip.basis(256, 240) + qutip.basis(256, 255))
magic_state.dag() * state
