from scipy.spatial.distance import hamming
from ..utils.logging_utils import get_formatted_logger
from ..utils.binary_string_utils import *
from ..utils.constraint_utils import get_independent_set_of_constraints
from ..utils.file_reading_utils import *
import numpy as np
from typing import List
import os

logger = get_formatted_logger(__name__)


def _change_i_bit(string: np.ndarray, i: int) -> np.ndarray:
    t1 = string.copy()
    t1[i] = (t1[i] + 1) % 2
    return t1


def _get_targets(n: int) -> List[List[np.ndarray]]:
    other_parity_strings = [
        item for sublist in
        [strings_with_weight(n, k) for k in range((n + 1) % 2, n, 2)]
        for item in sublist
    ]
    targets = [
        [other_parity_strings[i], other_parity_strings[j]]
        for i in range(len(other_parity_strings))
        for j in range(i + 1, len(other_parity_strings))
        if hamming(other_parity_strings[i], other_parity_strings[j]) * n > 2
    ]
    return targets


def _sorting_key(to_sort):
    """Used to sort nested lists by their first element"""
    return to_sort[0]


def _remove_duplicates(constraints: List[np.ndarray]) -> List[np.ndarray]:
    """Used to remove identical constraints from a list"""
    seen_elements = set()
    unique = []
    for constraint in constraints:
        sorted_constraint = tuple(map(tuple, sorted(constraint, key=_sorting_key)))
        if sorted_constraint not in seen_elements:
            unique.append(constraint)
            seen_elements.add(sorted_constraint)
    return unique


def _get_small_set_targets(n: int, offset: int) -> List[List[np.ndarray]]:
    other_parity_strings = [
        item for sublist in
        [strings_with_weight(n, k) for k in range((n + 1) % 2, n, 2)]
        for item in sublist
    ]
    # seen_elements = set()
    # targets = []
    # for i in range(len(other_parity_strings) - int(n * (n-1) / 2) - 1):
    #     valid_targets = [
    #         other_parity_strings[j]
    #         for j in range(len(other_parity_strings))
    #         if hamming(other_parity_strings[i], other_parity_strings[j]) * n > 2
    #         and tuple(list(other_parity_strings[j])) not in seen_elements
    #     ]
    #     targets.append([other_parity_strings[i], valid_targets[0]])
    #     seen_elements.add(tuple(list(valid_targets[0])))
    targets = [
        [other_parity_strings[offset], other_parity_strings[j]]
        for j in range(len(other_parity_strings))
        if hamming(other_parity_strings[offset], other_parity_strings[j]) * n > 2
    ]
    return targets


def _get_constraints_from_targets(targets: List[List[np.ndarray]]) -> List[np.ndarray]:
    constraints = []
    for target in targets:
        constraint = []
        for i in range(len(target[0])):
            if not target[0][i] == target[1][i]:
                constraint.append(
                    [read_binary_array(_change_i_bit(target[0], i)), read_binary_array(_change_i_bit(target[1], i))]
                )
        constraints.append(np.array(constraint))
    return constraints


def get_independent_constraints_directly(n: int) -> List[np.ndarray]:
    parity = n % 2
    if parity != 0:
        raise Exception('Only works for even n at the moment')

    targets = _get_targets(n)
    constraints = _get_constraints_from_targets(targets)
    independent_constraints = get_independent_set_of_constraints(constraints, n)

    filename = './data/IndependentConstraints/independent_constraints_%s.npy'
    directory_name = f'./data/IndependentConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    save_list_np_array(independent_constraints, filename % n)

    return independent_constraints


def get_independent_constraints_directly_from_small_set(n: int) -> List[np.ndarray]:
    parity = n % 2
    if parity != 0:
        raise Exception('Only works for even n at the moment')

    targets = _get_small_set_targets(n, 0)
    constraints = _get_constraints_from_targets(targets)
    independent_constraints = get_independent_set_of_constraints(constraints, n)

    filename = './data/IndependentConstraints/independent_constraints_small_set%s.npy'
    directory_name = f'./data/IndependentConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    save_list_np_array(independent_constraints, filename % n)

    return independent_constraints
