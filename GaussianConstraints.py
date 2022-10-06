import copy
import sympy
from GaussianStates import *
import numpy as np
import math


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


def get_highest_order_constraints(n, parity_of_targets: int):
    if parity_of_targets != 0 and parity_of_targets != 1:
        print('Parity must be either 0 or 1')
        exit()
    print(f'Starting for n = {n}')
    constraints_matrix = np.zeros((2 ** (n - 2), 2 ** n, 2))
    count = 0
    max_weight = 2 * math.floor((n + 2) / 4) if parity_of_targets == 1 else 2 * math.floor((n - 1) / 4)
    for k in range(parity_of_targets, max_weight, 2):
        targets = strings_with_weight(n, k)
        for target in targets:
            for i in range(n):
                x = copy.deepcopy(target)
                x[i] = (x[i] + 1) % 2
                y = (x + 1) % 2
                x_int = read_binary_array(x)
                y_int = read_binary_array(y)
                sorted_term = sorted([x_int, y_int])
                constraints_matrix[count, sorted_term[0], 0] = (-1) ** i
                constraints_matrix[count, sorted_term[1], 1] = 1
            count += 1

    return constraints_matrix


def get_lower_order_constraints(n, l):
    # l is number of bits where the tensor product of targets differ
    odd_weight_target_constraints = get_highest_order_constraints(l, 1)
    even_weight_target_constraints = get_highest_order_constraints(l, 0)
    new_constraints = []  # TODO make numpy

    for i in range(0, n - l + 1, 2):
        bit_strings = [0] * (n - l) if i == 0 else strings_with_weight(n - l, i)
        for string in bit_strings:
            new_constraints.append(odd_weight_target_constraints)

    return new_constraints


def verify_highest_order_constraints(n, constraints):
    N = 2 ** n
    gaussian = nn_gaussian_states(1, n, 0)
    for constraint in constraints:
        result = 0
        for i in range(len(constraint)):
            result += constraint[i] * gaussian[i] * gaussian[N - i - 1]
        np.testing.assert_almost_equal(result, 0)


for dim in range(6, 7):
    N = 2 ** dim
    constraints = get_highest_order_constraints(dim, 1)
    print(f'Number of constraints = {np.linalg.matrix_rank(constraints[:, :, 0])}')

    if dim < 8:
        _, indep_rows = sympy.Matrix(constraints[:, :, 0]).T.rref()
        unique_constraints = constraints[indep_rows, :, 0]
        verify_highest_order_constraints(dim, unique_constraints)
        print(unique_constraints)

    for number_of_differing_bits in range(4, dim - 1, 2):
        lower_order_constraints = get_lower_order_constraints(dim, number_of_differing_bits)
        print(lower_order_constraints)
# TODO: calculate the lower order constraints
