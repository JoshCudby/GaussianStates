import copy
from ..States.GaussianStates import *
from ..Utils.BinaryStringUtils import *
import numpy as np
import math


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
                constraints_matrix[count, sorted_term[1], 1] = 1  # this is not a good way to represent
            count += 1

    return constraints_matrix


def get_lower_order_constraints(n, l):
    # l is number of bits where the tensor product of targets differ
    odd_weight_target_constraints = get_highest_order_constraints(l, 1)
    even_weight_target_constraints = get_highest_order_constraints(l, 0)
    new_constraints = []  # TODO make numpy

    for i in range(0, n - l + 1, 2):
        bit_strings = [0] * (n - l) if i == 0 else strings_with_weight(n - l, i)
        # Need to find a way to correctly add the string where they agree to the constraint
        for string in bit_strings:
            for constraint in odd_weight_target_constraints:
                for j in range(2 ** (n - l)):
                    if constraint[j, 0] != 0:
                        first_label_bin = int_to_binary_array(j, n - l)
                        summed_first_label = np.array(np.meshgrid(string, first_label_bin))

                        second_label_bin = int_to_binary_array()
                        for summed_str in summed_first_label:
                            constraint_term_label = read_binary_array(summed_str)

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
