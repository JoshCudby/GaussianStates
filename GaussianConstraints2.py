from GaussianConstraints import strings_with_weight, read_binary_array
import numpy as np
import math


def int_to_binary_array(val: int, n: int) -> np.ndarray:
    return np.array([*"{0:0{1}b}".format(val, n)], dtype=int)


def get_highest_order_constraints(n: int) -> np.ndarray:
    print(f'Starting for n = {n}')
    constraints_matrix = np.zeros((2 ** (n - 2), n, 2))
    count = 0
    max_weight = 2 * math.floor((n + 2) / 4)
    for k in range(1, max_weight, 2):
        targets = strings_with_weight(n, k)
        for t in range(len(targets)):
            for i in range(n):
                x = targets[t]
                x[i] = (x[i] + 1) % 2
                y = (x + 1) % 2
                x_int = read_binary_array(x)
                y_int = read_binary_array(y)
                sorted_term = sorted([x_int, y_int])
                constraints_matrix[count, i, :] = sorted_term
            count += 1

    return constraints_matrix


def get_lower_order_constraints(constraints: np.ndarray) -> np.ndarray:
    return constraints


def get_all_constraints(n):
    parity = n % 2
    all_constraints = get_highest_order_constraints(4 + parity)
    for m in range(6 + parity, n + 1, 2):
        all_constraints = np.concatenate(get_highest_order_constraints(m), get_lower_order_constraints(all_constraints))
    return all_constraints


c = get_all_constraints(4)
print(4)
