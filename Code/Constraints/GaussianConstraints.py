from ..Utils.ConstraintUtils import *
from ..Utils.BinaryStringUtils import strings_with_weight, read_binary_array, int_to_binary_array
from ..Utils.FileReading import load_list_np_array, save_list_np_array
from ..Utils.Logging import get_formatted_logger
from ..States.GaussianStates import gaussian_states
import numpy as np
import math
from typing import List
import random

logger = get_formatted_logger(__name__)


def get_highest_order_constraints_even_case(n: int, parity_of_state: int) -> List[np.ndarray]:
    if n % 2 != 0:
        raise Exception('Only get highest order constraints for even n')
    if parity_of_state != 0 and parity_of_state != 1:
        raise Exception('Parity should be 0 or 1')

    constraints_list: List[np.ndarray] = [0] * (2 ** (n - 2))
    max_weight = 2 * math.floor((n + 2) / 4) if parity_of_state == 0 else 2 * math.floor(n / 4) + 1
    count = 0
    for k in range(1 - parity_of_state, max_weight, 2):
        targets = [[0] * n] if k == 0 else strings_with_weight(n, k)

        if k == n / 2:
            targets = targets[0:int(len(targets) / 2)]

        for target in targets:
            constraint = np.zeros((n, 2), dtype=int)
            for i in range(n):
                x = np.copy(target)
                x[i] = (x[i] + 1) % 2
                y = (x + 1) % 2
                x_int = read_binary_array(x)
                y_int = read_binary_array(y)
                sorted_term = sorted([x_int, y_int])
                constraint[i, :] = sorted_term
            constraints_list[count] = constraint
            count += 1
    return constraints_list


def get_lower_order_constraints(constraints: List[np.ndarray], m: int) -> List[np.ndarray]:
    all_new_constraints = []
    # Loop over: constraints, even weight strings of length 2, choices of how to position
    for constraint in constraints:
        constraint_in_binary = [[int_to_binary_array(int(t), m - 2) for t in term] for term in constraint]
        length = len(constraint_in_binary[0][0])
        new_constraints_in_binary = [
            [[np.insert(binary_array, [i, j], b) for binary_array in term] for term in constraint_in_binary]
            for i in range(length + 1)
            for j in range(i, length + 1)
            for b in [[0, 0], [1, 1]]
        ]
        new_constraints = [
            [[read_binary_array(binary_array) for binary_array in term] for term in constraint]
            for constraint in new_constraints_in_binary
        ]
        for new_constraint in new_constraints:
            if len(new_constraint) > 0:
                all_new_constraints.append(np.array(new_constraint, dtype=int))

    all_new_constraints = remove_duplicates(all_new_constraints)
    return all_new_constraints


def sorting_key(to_sort):
    return to_sort[0]


def remove_duplicates(constraints: List[np.ndarray]) -> List[np.ndarray]:
    # This removes identical elements
    seen_elements = set()
    unique = []
    for constraint in constraints:
        sorted_constraint = tuple(map(tuple, sorted(constraint, key=sorting_key)))
        if sorted_constraint not in seen_elements:
            unique.append(constraint)
            seen_elements.add(sorted_constraint)
    return unique


def get_highest_order_constraints_odd_case(n: int) -> List[np.ndarray]:
    all_new_constraints = []
    even_parity_constraints = remove_duplicates(get_highest_order_constraints_even_case(n - 1, 0))
    odd_parity_constraints = remove_duplicates(get_highest_order_constraints_even_case(n - 1, 1))

    # Loop over: constraints, strings of length 1, choices of how to position
    for constraint in even_parity_constraints:
        constraint_in_binary = [[int_to_binary_array(int(t), n - 1) for t in term] for term in constraint]
        length = len(constraint_in_binary[0][0])
        new_constraints_in_binary = [
            [[np.insert(binary_array, i, 0) for binary_array in term] for term in constraint_in_binary]
            for i in range(length + 1)
        ]
        new_constraints = [
            [[read_binary_array(binary_array) for binary_array in term] for term in constraint]
            for constraint in new_constraints_in_binary
        ]
        for new_constraint in new_constraints:
            all_new_constraints.append(np.array(new_constraint))

    for constraint in odd_parity_constraints:
        constraint_in_binary = [[int_to_binary_array(int(t), n - 1) for t in term] for term in constraint]
        length = len(constraint_in_binary[0][0])
        new_constraints_in_binary = [
            [[np.insert(binary_array, i, 1) for binary_array in term] for term in constraint_in_binary]
            for i in range(length + 1)
        ]
        new_constraints = [
            [[read_binary_array(binary_array) for binary_array in term] for term in constraint]
            for constraint in new_constraints_in_binary
        ]
        for new_constraint in new_constraints:
            all_new_constraints.append(np.array(new_constraint))

    return all_new_constraints


def get_all_constraints(n: int) -> List[np.ndarray]:
    filename = './Output/Constraints/all_constraints_%s.npy'
    if n < 4:
        return []
    parity = n % 2

    all_constraints = []
    x = n
    while x > 3:
        try:
            all_constraints = load_list_np_array(filename % x)
            logger.info(f'Loaded constraints for n = {x}')
            break
        except FileNotFoundError:
            x -= 2

    if len(all_constraints) == 0:
        x += 2

    if parity == 0:
        if x == 4 and len(all_constraints) == 0:
            all_constraints = remove_duplicates(get_highest_order_constraints_even_case(4, 0))
            # all_constraints = get_highest_order_constraints_even_case(4, 0)
            logger.info(f'Saving constraints for n = 4')
            save_list_np_array(all_constraints, filename % 4)
        for m in range(x + 2, n + 1, 2):
            all_constraints = get_highest_order_constraints_even_case(m, 0) \
                              + get_lower_order_constraints(all_constraints, m)
            state = gaussian_states(1, m)
            verify_constraints(all_constraints, state)
            logger.info(f'Saving constraints for n = {m}')
            save_list_np_array(all_constraints, filename % m)
            # logger.info(f'For n={m}, there are {len(all_constraints)} constraints')
    else:
        if x == 5 and len(all_constraints) == 0:
            all_constraints = remove_duplicates(get_highest_order_constraints_odd_case(5))
            save_list_np_array(all_constraints, filename % 5)
        for m in range(x + 2, n + 1, 2):
            all_constraints = get_highest_order_constraints_odd_case(m) \
                              + get_lower_order_constraints(all_constraints, m)
            state = gaussian_states(1, m)
            verify_constraints(all_constraints, state)
            logger.info(f'Saving constraints for n = {m}')
            save_list_np_array(all_constraints, filename % m)
            # logger.info(f'For n={m}, there are {len(all_constraints)} constraints')
    return all_constraints


def get_independent_constraints(all_constraints: List[np.ndarray], state: np.ndarray) -> List[np.ndarray]:
    independent_constraints = []

    x_values_set = set()
    for z in range(len(all_constraints)):
        test_constraints = independent_constraints.copy()
        test_constraints.append(all_constraints[z])
        m = len(test_constraints)

        flattened_constraints = [constraint.flatten() for constraint in test_constraints]
        a_labels = [item for sublist in flattened_constraints for item in sublist]
        random.shuffle(a_labels)
        # a_labels = sorted(a_labels)  # Same number of constraints for both of these options, but this is slower

        test_x_values = list(x_values_set.copy())
        for a in a_labels:
            if a not in x_values_set:
                test_x_values.append(a)
                break

        if m > len(test_x_values):
            raise Exception('Not enough a values')
        jacobian = np.zeros((m, m), dtype=complex)
        for i in range(m):
            constraint = test_constraints[i]
            for j in range(m):
                x = test_x_values[j]
                for constraint_index in range(len(constraint)):
                    constraint_term = constraint[constraint_index]
                    for index in range(2):
                        if x == constraint_term[index]:
                            label_to_add = constraint_term[(index + 1) % 2]
                            jacobian[i, j] = complex(state[label_to_add]) * ((-1) ** constraint_index)

        # sv = np.linalg.svd(jacobian, compute_uv=False)
        rank = np.linalg.matrix_rank(jacobian)
        if rank == m:
            independent_constraints.append(all_constraints[z])
            x_values_set = set(test_x_values)
        # if z % 1000 == 0:
        #     print(f'Constraint number {z} reached')

    return independent_constraints


def get_number_highest_order_independent_constraints(constraints: List[np.ndarray], state: np.ndarray, dim: int) -> int:
    independent_constraints = get_independent_constraints(constraints, state)
    return len([x for x in independent_constraints if len(x) == dim])


def get_constraints_seen_for_targets(
        all_constraints: List[np.ndarray],
        indexes: List[int],
        state: np.ndarray,
        dim: int,
        number_of_runs: int
) -> set:
    all_seen_constraints = set()
    targets = [tuple(map(tuple, constraint)) for constraint in [all_constraints[x] for x in indexes]]
    for count in range(number_of_runs):
        random.shuffle(all_constraints)
        independent_constraints = get_independent_constraints(all_constraints, state)
        # print(f'Number of independent constraints = {len(independent_constraints)}')

        long_constraints = [constraint for constraint in independent_constraints if len(constraint) == dim]
        # print(f'{len(long_constraints)} highest order constraints')

        to_add = [False] * len(targets)
        for constraint in long_constraints:
            mapped = tuple(map(tuple, constraint))
            for index, t in enumerate(targets):
                if t == mapped:
                    to_add[index] = True
        if all(to_add):
            for constraint in long_constraints:
                all_seen_constraints.add(tuple(map(tuple, constraint)))
    return all_seen_constraints


def get_matrix_of_independent_constraint_possibilities(
        all_constraints: List[np.ndarray],
        number_of_runs: int,
        dim: int
) -> np.ndarray:
    matrix = np.zeros((number_of_runs, len(all_constraints)), dtype=int)
    for i in range(number_of_runs):
        state = gaussian_states(1, dim)
        independent_constraints = get_independent_constraints(all_constraints, state)
        independent_constraints = [tuple(map(tuple, constraint)) for constraint in independent_constraints]
        for column, constraint in enumerate(all_constraints):
            c = tuple(map(tuple, constraint))
            for independent_constraint in independent_constraints:
                if c == independent_constraint:
                    matrix[i, column] = 1
    return matrix


def get_independent_constraints_for_next_order(
        independent_constraints: List[np.ndarray],
        n: int,
        filename: str = None
) -> List[np.ndarray]:
    new_highest_order_constraints = get_highest_order_constraints_even_case(n, 0)
    mapped_existing_constraints = get_lower_order_constraints(independent_constraints, n)
    return get_independent_set_of_constraints(new_highest_order_constraints + mapped_existing_constraints, n, filename)
