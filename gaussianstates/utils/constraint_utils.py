import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from typing import List
from .binary_string_utils import strings_with_weight, read_binary_array, change_i_bit
from .logging_utils import get_formatted_logger
from time import time
from .file_reading_utils import save_list_np_array
from gaussianstates.states.gaussian_states import gaussian_states
from scipy.spatial.distance import hamming

logger = get_formatted_logger(__name__)


def verify_constraints(constraints: List[np.ndarray], state: np.ndarray) -> None:
    """Check that a state satisfies a list of constraints"""
    for cons in constraints:
        val = 0
        for j in range(len(cons)):
            term = cons[j]
            val += ((-1) ** j) * state[term[0]] * state[term[1]]
        if abs(val) > 10 ** (-12):
            logger.error(cons)
            logger.error(val)
            raise Exception('Constraint not satisfied')


def get_targets(n: int) -> List[List[np.ndarray]]:
    odd_parity_strings = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(1, n, 2)]
        for item in sublist
    ]
    targets = [
        [odd_parity_strings[i], odd_parity_strings[j]]
        for i in range(len(odd_parity_strings))
        for j in range(i + 1, len(odd_parity_strings))
        if hamming(odd_parity_strings[i], odd_parity_strings[j]) * n > 2
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
    return remove_duplicates(constraints)


def _sorting_key(to_sort):
    """Used to sort nested lists by their first element"""
    return min(to_sort)


def remove_duplicates(constraints: List[np.ndarray]) -> List[np.ndarray]:
    """Used to remove identical constraints from a list"""
    seen_constraint_labels = set()
    unique = []
    for constraint in constraints:
        constraint_labels = tuple(sorted([t for term in constraint for t in term]))
        if not any([
            len(seen) == len(constraint_labels) and
            all([
                constraint_labels[i] == seen[i] for i in range(len(constraint_labels))
            ]) for seen in seen_constraint_labels
        ]):
            unique.append(constraint)
            seen_constraint_labels.add(constraint_labels)
    return unique


def make_jacobian(constraints: List[np.ndarray], x_values: List[int], n: int):
    state = gaussian_states(1, n)
    J = np.zeros((len(constraints), len(x_values)), dtype=complex)
    for i in range(len(constraints)):
        for j in range(len(x_values)):
            J[i][j] = _differentiate_constraint_with_state(constraints[i], x_values[j], state)
    return J


def _differentiate_constraint_with_state(constraint: np.ndarray, x_value: int, state: np.ndarray):
    """Differentiate a constraint with respect to a given variable, and evaluate at the state.

    Arguments:
    constraint -- an independent constraint set for n qubits
    x_values -- the value to differentiate with respect to
    state -- a gaussian state of n qubits which will be used to evaluate the Jacobian for implicit function theorem
    """
    return np.sum([
        complex(state[constraint[constraint_index][(index + 1) % 2]] * (-1) ** constraint_index)
        if constraint[constraint_index][index] == x_value else 0
        for index in range(2)
        for constraint_index in range(len(constraint))])


def form_jacobian(
        jacobian: np.ndarray,
        independent_constraints: List[np.ndarray],
        test_constraint: np.ndarray,
        x_values: List[int],
        test_x_value: int,
        state: np.ndarray,
        m: int
):
    # Handle the special case for the first loop
    if jacobian is None:
        value = _differentiate_constraint_with_state(test_constraint, test_x_value, state)
        new_jacobian = np.array([[value]])
        x_values = [test_x_value]
    else:
        column_to_add = [
            _differentiate_constraint_with_state(constraint, test_x_value, state)
            for constraint in independent_constraints
        ]
        x_values = x_values + [test_x_value]
        row_to_add = [
            _differentiate_constraint_with_state(test_constraint, x_values[j], state)
            for j in range(m)
        ]

        new_jacobian = np.column_stack((jacobian, column_to_add))
        new_jacobian = np.vstack((new_jacobian, row_to_add))

    # Can this be optimized?
    rank = np.linalg.matrix_rank(new_jacobian)
    if rank == m:
        return [independent_constraints + [test_constraint], x_values, new_jacobian]
    return None


def _independent_constraint_iteration(
        independent_constraints: List[np.ndarray],
        test_constraint: np.ndarray,
        x_values: List[int],
        state: np.ndarray,
        jacobian: np.ndarray = None,
):
    """Test whether a constraint is independent of a given set.

    Arguments:
    independent_constraints -- an independent constraint set for n qubits
    test_constraint -- a speculative new member of the set
    x_values -- the values which will be used as independent for the implicit function theorem
    state -- a gaussian state of n qubits which will be used to evaluate the Jacobian for implicit function theorem
    jacobian -- the Jacobian for the independent set
    """
    m = len(independent_constraints) + 1

    new_a_labels = test_constraint.flatten()
    new_test_x_value = None

    for term in test_constraint:
        for index, t in enumerate(term):
            if t == 0:
                a = term[(index + 1) % 2]
                if a not in x_values:
                    new_test_x_value = a
                    return form_jacobian(
                        jacobian, independent_constraints, test_constraint, x_values, new_test_x_value, state, m
                    )

    if new_test_x_value is None:
        for a in new_a_labels:
            if a not in x_values:
                return form_jacobian(
                    jacobian, independent_constraints, test_constraint, x_values, a, state, m
                )
    return None


def get_independent_set_of_constraints(
        constraints: List[np.ndarray],
        n: int,
        filename: str = None
) -> (List[np.ndarray], List[int]):
    """Find the independent constraints from a set of constraints on n qubits.

    Arguments:
    constraints -- constraints for n qubits
    n -- the number of qubits
    filename -- where to save the output (default = None)
    """
    independent_constraints = []
    state = gaussian_states(1, n)
    x_values = []
    jacobian = None
    independent_constraint_counter = 0
    start_time = time()

    for z in range(len(constraints)):
        if time() - start_time > 300:
            logger.info(f'Currently on iteration {z} out of {len(constraints)}.'
                        f'\n Current matrix size is {len(independent_constraints)}.')
            if independent_constraint_counter == len(independent_constraints) and filename is not None:
                logger.info(f'Matrix size has not changed. Saving ...')
                save_list_np_array(independent_constraints, filename)
            independent_constraint_counter = len(independent_constraints)
            start_time = time()
        results = _independent_constraint_iteration(
            independent_constraints,
            constraints[z],
            x_values,
            state,
            jacobian
        )
        if results is not None:
            independent_constraints, x_values, jacobian = results

    return independent_constraints, x_values


def get_independent_set_of_constraints_mp(
        constraints: List[np.ndarray],
        n: int,
        independent_constraints: List[np.ndarray] = None,
        x_values: List[int] = None,
        jacobian: np.ndarray = None,
        state: np.ndarray = None,
        filename: str = None,
        number_of_cores: int = 4
) -> List[np.ndarray]:
    """Find the independent constraints from a set of constraints on n qubits.

    Arguments:
    constraints -- constraints for n qubits
    n -- the number of qubits
    filename -- where to save the output (default = None)
    number_of_cores -- how many workers are available (default = 4)
    """
    if x_values is None:
        x_values = []
    if independent_constraints is None:
        independent_constraints = []
    if state is None:
        state = gaussian_states(1, n)
    independent_constraint_counter = len(independent_constraints)
    script_start_time = time()
    iteration_start_time = time()
    did_add_previous = [1] * number_of_cores
    while len(constraints) > 0:
        if (time() - script_start_time) / (60 * 60) > 11:
            logger.info('Script about to end. Saving partial data...')
            save_list_np_array(independent_constraints, filename)
            base_filename = filename.split('.')[0]

            save_list_np_array(constraints, base_filename + '_all')
            np.save(base_filename + '_x', x_values)
            np.save(base_filename + '_J', jacobian)
            np.save(base_filename + '_state', state)

        if time() - iteration_start_time > 300:
            logger.info(f'Currently {len(constraints)} remaining.'
                        f'\n Current matrix size is {len(independent_constraints)}.')
            if independent_constraint_counter == len(independent_constraints) and filename is not None:
                logger.info(f'Matrix size has not changed. Saving ...')
                save_list_np_array(independent_constraints, filename)
            independent_constraint_counter = len(independent_constraints)
            iteration_start_time = time()

        # Batch running is very slow near the start, when almost every constraint is independent
        # Need some heuristic for when to start batching
        if np.mean(did_add_previous) > 0.5:
            result = _independent_constraint_iteration(
                independent_constraints,
                constraints.pop(),
                x_values,
                state,
                jacobian
            )
            if result is not None:
                x_values = result[1]
                jacobian = result[2]
                did_add_previous.pop(0)
                did_add_previous.append(1)
            else:
                did_add_previous.pop(0)
                did_add_previous.append(0)

        else:
            with Pool(number_of_cores) as pool:
                results = pool.map(
                    _independent_constraint_iteration,
                    [independent_constraints] * number_of_cores,
                    [constraints[-z] for z in range(1, min(number_of_cores, len(constraints)) + 1)],
                    [x_values] * number_of_cores,
                    [state] * number_of_cores,
                    [jacobian] * number_of_cores
                )

            have_added_constraint = False
            constraints_to_retry = []
            for result in results:
                did_add_previous.pop(0)
                if result is None:
                    constraints.pop()
                    did_add_previous.append(0)
                else:
                    did_add_previous.append(1)
                    if not have_added_constraint:
                        independent_constraints.append(constraints.pop())
                        x_values = result[1]
                        jacobian = result[2]
                        have_added_constraint = True
                    else:
                        constraints_to_retry.append(constraints.pop())
            constraints += constraints_to_retry

    return independent_constraints
