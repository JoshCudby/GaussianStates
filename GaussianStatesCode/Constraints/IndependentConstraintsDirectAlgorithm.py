from scipy.spatial.distance import hamming
from ..Utils.Logging import get_formatted_logger
from ..Utils.BinaryStringUtils import *
from ..Utils.ConstraintUtils import get_independent_set_of_constraints
from ..Utils.FileReading import *
import numpy as np
from typing import List
import os

logger = get_formatted_logger(__name__)


def change_i_bit(string: np.ndarray, i: int) -> np.ndarray:
    t1 = string.copy()
    t1[i] = (t1[i] + 1) % 2
    return t1


def get_targets(n: int) -> List[List[np.ndarray]]:
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


def get_constraints_from_targets(targets: List[List[np.ndarray]]) -> List[np.ndarray]:
    constraints = []
    for target in targets:
        constraint = []
        for i in range(len(target[0])):
            if not target[0][i] == target[1][i]:
                constraint.append(
                    [read_binary_array(change_i_bit(target[0], i)), read_binary_array(change_i_bit(target[1], i))]
                )
        constraints.append(np.array(constraint))
    return constraints


def get_independent_constraints_directly(n: int) -> List[np.ndarray]:
    parity = n % 2
    if parity != 0:
        raise Exception('Only works for even n at the moment')

    targets = get_targets(n)
    constraints = get_constraints_from_targets(targets)
    independent_constraints = get_independent_set_of_constraints(constraints, n)

    filename = './Output/IndependentConstraints/independent_constraints_%s.npy'
    directory_name = f'./Output/IndependentConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    save_list_np_array(independent_constraints, filename % n)

    return independent_constraints
