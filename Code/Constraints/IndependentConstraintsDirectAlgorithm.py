from scipy.spatial.distance import hamming
from ..Utils.Logging import get_formatted_logger
from ..Utils.BinaryStringUtils import *
from ..States.GaussianStates import gaussian_states
from ..Utils.FileReading import *
import numpy as np
from typing import List
import random
import os

logger = get_formatted_logger(__name__)


def change_i_bit(string: np.ndarray, i: int) -> np.ndarray:
    t1 = string.copy()
    t1[i] = (t1[i] + 1) % 2
    return t1


def get_targets(n: int) -> List[List[np.ndarray]]:
    other_parity_strings = [item for sublist in
                            [strings_with_weight(n, k) for k in range((n + 1) % 2, n, 2)]
                            for item in sublist]
    targets = [[other_parity_strings[i], other_parity_strings[j]]
               for i in range(len(other_parity_strings))
               for j in range(i + 1, len(other_parity_strings))
               if hamming(other_parity_strings[i], other_parity_strings[j]) * n > 2]
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


def get_independent_set_of_constraints(constraints: List[np.ndarray], n: int):
    independent_constraints = []
    state = gaussian_states(1, n)

    x_values_set = set()
    for z in range(len(constraints)):
        test_constraints = independent_constraints.copy()
        test_constraints.append(constraints[z])
        m = len(test_constraints)

        new_a_labels = constraints[z].flatten()
        test_x_values = list(x_values_set.copy())
        for a in new_a_labels:
            if a not in x_values_set:
                test_x_values.append(a)
                break

        if m > len(test_x_values):
            flattened_constraints = [constraint.flatten() for constraint in test_constraints]
            a_labels = [item for sublist in flattened_constraints for item in sublist]
            # random.shuffle(a_labels)
            for a in a_labels:
                if a not in x_values_set:
                    test_x_values.append(a)
                    break

        if m > len(test_x_values):
            raise Exception('Not enough a values')

        jacobian = np.array(
            [[np.sum([
                complex(state[constraint[constraint_index][(index + 1) % 2]] * (-1) ** constraint_index)
              if constraint[constraint_index][index] == test_x_values[j] else 0
                for index in range(2)
                for constraint_index in range(len(constraint))])
                for j in range(m)]
                for constraint in test_constraints],
            dtype=complex
        )

        rank = np.linalg.matrix_rank(jacobian)
        if rank == m:
            independent_constraints.append(constraints[z])
            x_values_set = set(test_x_values)
    return independent_constraints


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
