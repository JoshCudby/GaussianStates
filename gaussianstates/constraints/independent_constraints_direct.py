import random
from scipy.spatial.distance import hamming
from ..utils.logging_utils import get_formatted_logger
from ..utils.binary_string_utils import *
from ..utils.constraint_utils import get_independent_set_of_constraints, make_jacobian, get_targets, \
    get_constraints_from_targets
from ..utils.file_reading_utils import *
import numpy as np
from typing import List
import os

logger = get_formatted_logger(__name__)


def _sorting_key(to_sort):
    """Used to sort nested lists by their first element"""
    return to_sort[0]


def get_small_set_targets(n: int) -> List[List[np.ndarray]]:
    if not n % 2 == 0:
        raise Exception('Even n only')
    odd_parity_strings = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(1, n, 2)]
        for item in sublist
    ]

    targets = []
    zero_weight_string = [0] * n
    for i in range(0, n - 2):
        first_target = change_i_bit(zero_weight_string, i)
        second_targets = [
            odd_parity_strings[j]
            for j in range(len(odd_parity_strings))
            if hamming(first_target, odd_parity_strings[j]) * n > 2
            and all([odd_parity_strings[j][k] == 0 for k in range(i + 1)])
        ]
        for second_target in second_targets:
            targets.append([first_target, second_target])

    return targets


def get_independent_constraints_directly(n: int) -> List[np.ndarray]:
    parity = n % 2
    if parity != 0:
        raise Exception('Only works for even n at the moment')

    targets = get_targets(n)
    constraints = get_constraints_from_targets(targets)
    independent_constraints = get_independent_set_of_constraints(constraints, n)

    filename = './data/IndependentConstraints/independent_constraints_%s.npy'
    directory_name = f'./data/IndependentConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    save_list_np_array(independent_constraints, filename % n)

    return independent_constraints


def get_independent_constraints_directly_from_small_set(n: int) -> (List[np.ndarray], np.ndarray, List[int]):
    parity = n % 2
    if parity != 0:
        raise Exception('Only works for even n at the moment')

    targets = get_small_set_targets(n)
    constraints = get_constraints_from_targets(targets)
    independent_constraints, _ = get_independent_set_of_constraints(constraints, n)

    even_weight = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(4, n + 1, 2)]
        for item in sublist
    ]

    ints = [read_binary_array(i) for i in even_weight]
    J = make_jacobian(independent_constraints, ints, n)

    filename = 'data/IndependentConstraints/independent_constraints_small_set%s.npy'
    directory_name = 'data/IndependentConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    save_list_np_array(independent_constraints, filename % n)

    return independent_constraints, J, ints


def get_independent_constraints_directly_from_target_set(
    targets: List[List[np.ndarray]],
    n: int
):
    parity = n % 2
    if parity != 0:
        raise Exception('Only works for even n at the moment')

    constraints = get_constraints_from_targets(targets)
    independent_constraints = get_independent_set_of_constraints(constraints, n)

    # even_weight = [
    #     item for sublist in
    #     [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
    #     for item in sublist
    # ]
    #
    # ints_bin = random.sample(even_weight, 16)
    # ints = [read_binary_array(i) for i in ints_bin]
    # J = make_jacobian(independent_constraints, ints, n)
    #
    # distances = []
    #
    # for e in even_weight:
    #     count = 0
    #     for arr in ints_bin:
    #         if hamming(e, arr) * n > 2:
    #             count += 1
    #     distances.append(count)
    # d_max = max(distances)

    filename = 'data/IndependentConstraints/independent_constraints_small_set_2%s.npy'
    directory_name = 'data/IndependentConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    save_list_np_array(independent_constraints, filename % n)

    return independent_constraints  # , J, ints, d_max
