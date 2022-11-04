import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from typing import List
from .logging_utils import get_formatted_logger
from time import time
from ..states.gaussian_states import gaussian_states
from .file_reading_utils import save_list_np_array

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
        value = _differentiate_constraint_with_state(test_constraint, new_test_x_value, state)
        new_jacobian = np.array([[value]])
    else:
        column_to_add = [
            _differentiate_constraint_with_state(constraint, new_test_x_value, state)
            for constraint in independent_constraints
        ]
        row_to_add = [
            _differentiate_constraint_with_state(test_constraint, test_x_values[j], state)
            for j in range(m)
        ]

        new_jacobian = np.column_stack((jacobian, column_to_add))
        new_jacobian = np.vstack((new_jacobian, row_to_add))

    # Can this be optimized?
    rank = np.linalg.matrix_rank(new_jacobian)
    if rank == m:
        independent_constraints.append(test_constraint)
        x_values = test_x_values
        jacobian = new_jacobian
        return [independent_constraints, x_values, jacobian]
    return None


def get_independent_set_of_constraints(constraints: List[np.ndarray], n: int, filename: str = None) -> List[np.ndarray]:
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
        if all(results):
            independent_constraints, x_values, jacobian = results

    return independent_constraints


def get_independent_set_of_constraints_mp(
        constraints: List[np.ndarray],
        n: int,
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
    independent_constraints = []
    state = gaussian_states(1, n)
    x_values = []
    jacobian = None
    independent_constraint_counter = 0
    start_time = time()
    did_add_previous = [1] * number_of_cores
    while len(constraints) > 0:
        if time() - start_time > 300:
            logger.info(f'Currently {len(constraints)} remaining.'
                        f'\n Current matrix size is {len(independent_constraints)}.')
            if independent_constraint_counter == len(independent_constraints) and filename is not None:
                logger.info(f'Matrix size has not changed. Saving ...')
                save_list_np_array(independent_constraints, filename)
            independent_constraint_counter = len(independent_constraints)
            start_time = time()

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
