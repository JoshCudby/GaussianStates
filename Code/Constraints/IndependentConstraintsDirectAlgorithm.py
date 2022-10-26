from scipy.spatial.distance import hamming
from ..Utils.Logging import get_formatted_logger
from ..Utils.BinaryStringUtils import *
from ..States.GaussianStates import gaussian_states
from ..Utils.FileReading import *
import numpy as np
from typing import List
from time import time
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


def differentiate_constraint_with_state(constraint: np.ndarray, x_value: int, state: np.ndarray):
    return np.sum([
        complex(state[constraint[constraint_index][(index + 1) % 2]] * (-1) ** constraint_index)
        if constraint[constraint_index][index] == x_value else 0
        for index in range(2)
        for constraint_index in range(len(constraint))])


def get_independent_set_of_constraints(constraints: List[np.ndarray], n: int):
    start_time = time()
    independent_constraints = []
    state = gaussian_states(1, n)
    x_values = []
    jacobian = None
    for z in range(len(constraints)):
        if time() - start_time > 30:
            logger.info(f'Currently on iteration {z} out of {len(constraints)}.'
                        f'\n Current matrix size is {len(independent_constraints)}.')
            start_time = time()
        new_test_constraint = constraints[z]
        m = len(independent_constraints) + 1

        new_a_labels = new_test_constraint.flatten()
        test_x_values = None
        new_test_x_value = None

        for a in new_a_labels:
            if a not in x_values:
                new_test_x_value = a
                test_x_values = x_values + [a]
                break

        if new_test_x_value is None:
            flattened_constraints = [constraint.flatten() for constraint in independent_constraints]
            a_labels = [item for sublist in flattened_constraints for item in sublist]
            for a in a_labels:
                if a not in x_values:
                    new_test_x_value = a
                    test_x_values = x_values + [a]
                    break

        if new_test_x_value is None:
            raise Exception('Could not find a new a label to differentiate w.r.t.')

        # Handle the special case for the first loop
        if jacobian is None:
            value = differentiate_constraint_with_state(new_test_constraint, new_test_x_value, state)
            new_jacobian = np.array([[value]])
        else:
            column_to_add = [
                differentiate_constraint_with_state(constraint, new_test_x_value, state)
                for constraint in independent_constraints
            ]
            new_jacobian = np.column_stack((jacobian, column_to_add))

            row_to_add = [
                differentiate_constraint_with_state(new_test_constraint, test_x_values[j], state)
                for j in range(m)
            ]
            new_jacobian = np.vstack((new_jacobian, row_to_add))

        rank = np.linalg.matrix_rank(new_jacobian)
        if rank == m:
            independent_constraints.append(new_test_constraint)
            x_values = test_x_values
            jacobian = new_jacobian
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
