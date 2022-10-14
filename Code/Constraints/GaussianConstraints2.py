from ..Utils.BinaryStringUtils import strings_with_weight, read_binary_array, int_to_binary_array
import numpy as np
import math


def get_highest_order_constraints_even_case(n: int, parity_of_state: int) -> list[np.ndarray]:
    if n % 2 != 0:
        raise Exception('Only get highest order constraints for even n')
    if parity_of_state != 0 and parity_of_state != 1:
        raise Exception('Parity should be 0 or 1')

    constraints_list: list[np.ndarray] = [0] * (2 ** (n - 2))
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


def get_lower_order_constraints(constraints: list[np.ndarray], m: int) -> list[np.ndarray]:
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
            all_new_constraints.append(np.array(new_constraint))
    return all_new_constraints


def sorting_key(to_sort):
    return to_sort[0]


def remove_duplicates(constraints: list[np.ndarray]) -> list[np.ndarray]:
    # This removes identical elements, but it would be nice to be able to remove constraints which don't reduce the DoF
    seen_elements = set()
    unique = []
    for constraint in constraints:
        sorted_constraint = tuple(map(tuple, sorted(constraint, key=sorting_key)))
        if sorted_constraint not in seen_elements:
            unique.append(constraint)
            seen_elements.add(sorted_constraint)
        # else:
        #     print(constraint)
    return unique


def get_highest_order_constraints_odd_case(n: int) -> list[np.ndarray]:
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


def get_all_constraints(n: int) -> list[np.ndarray]:
    if n < 4:
        return []
    parity = n % 2

    if parity == 0:
        all_constraints = remove_duplicates(get_highest_order_constraints_even_case(4, 0))
        for m in range(6, 2 * math.floor(n / 2) + 1, 2):
            all_constraints = get_highest_order_constraints_even_case(m, 0) \
                              + remove_duplicates( get_lower_order_constraints(all_constraints, m))
            # print(f'For n={m}, there are {len(all_constraints)} constraints')
    else:
        c = get_highest_order_constraints_odd_case(5)
        all_constraints = remove_duplicates(c)
        for m in range(7, n + 1, 2):
            all_constraints = get_highest_order_constraints_odd_case(m) \
                              + remove_duplicates(get_lower_order_constraints(all_constraints, m))
            # print(f'For n={m}, there are {len(all_constraints)} constraints')
    return all_constraints
