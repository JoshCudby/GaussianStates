import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from typing import List
from .Logging import get_formatted_logger
from time import time
from ..States.GaussianStates import gaussian_states
from .FileReading import save_list_np_array

logger = get_formatted_logger(__name__)


def verify_constraints(constraints: List[np.ndarray], state: np.ndarray) -> None:
    for cons in constraints:
        val = 0
        for j in range(len(cons)):
            term = cons[j]
            val += ((-1) ** j) * state[term[0]] * state[term[1]]
        if abs(val) > 10 ** (-12):
            logger.error(cons)
            logger.error(val)
            raise Exception('Constraint not satisfied')


def differentiate_constraint_with_state(constraint: np.ndarray, x_value: int, state: np.ndarray):
    return np.sum([
        complex(state[constraint[constraint_index][(index + 1) % 2]] * (-1) ** constraint_index)
        if constraint[constraint_index][index] == x_value else 0
        for index in range(2)
        for constraint_index in range(len(constraint))])


def independent_constraint_iteration(
    independent_constraints: List[np.ndarray],
    new_test_constraint: np.ndarray,
    x_values: List[int],
    state: np.ndarray,
    jacobian: np.ndarray = None,
):
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
        # These could be parallelized
        column_to_add = [
            differentiate_constraint_with_state(constraint, new_test_x_value, state)
            for constraint in independent_constraints
        ]
        row_to_add = [
            differentiate_constraint_with_state(new_test_constraint, test_x_values[j], state)
            for j in range(m)
        ]

        new_jacobian = np.column_stack((jacobian, column_to_add))
        new_jacobian = np.vstack((new_jacobian, row_to_add))

    # Can this be optimized?
    rank = np.linalg.matrix_rank(new_jacobian)
    if rank == m:
        independent_constraints.append(new_test_constraint)
        x_values = test_x_values
        jacobian = new_jacobian
        return [independent_constraints, x_values, jacobian]
    return [None, None, None]


def get_independent_set_of_constraints(constraints: List[np.ndarray], n: int, filename: str = None) -> List[np.ndarray]:
    independent_constraints = []
    state = gaussian_states(1, n)
    x_values = []
    jacobian = None
    independent_constraint_counter = 0
    start_time = time()

    # Could batch run many of these with MPI, but would need some careful synchronization
    for z in range(len(constraints)):
        if time() - start_time > 300:
            logger.info(f'Currently on iteration {z} out of {len(constraints)}.'
                        f'\n Current matrix size is {len(independent_constraints)}.')
            if independent_constraint_counter == len(independent_constraints) and filename is not None:
                logger.info(f'Matrix size has not changed. Saving ...')
                save_list_np_array(independent_constraints, filename)
            independent_constraint_counter = len(independent_constraints)
            start_time = time()
        results = independent_constraint_iteration(
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
    filename: str = None
) -> List[np.ndarray]:
    independent_constraints = []
    state = gaussian_states(1, n)
    x_values = []
    jacobian = None
    independent_constraint_counter = 0
    start_time = time()
    number_cpus = 18

    while len(constraints) > 0:
        if time() - start_time > 300:
            logger.info(f'Currently {len(constraints)} remaining.'
                        f'\n Current matrix size is {len(independent_constraints)}.')
            if independent_constraint_counter == len(independent_constraints) and filename is not None:
                logger.info(f'Matrix size has not changed. Saving ...')
                save_list_np_array(independent_constraints, filename)
            independent_constraint_counter = len(independent_constraints)
            start_time = time()

        with Pool(number_cpus) as pool:
            results = pool.map(
                independent_constraint_iteration,
                [independent_constraints] * number_cpus,
                [constraints[-z] for z in range(number_cpus)],
                [x_values] * number_cpus,
                [state] * number_cpus,
                [jacobian] * number_cpus
            )

        have_added_constraint = False
        constraints_to_retry = []
        for result in results:
            if not all(result):
                constraints.pop()
                logger.info('Not adding this constraint')
            else:
                if not have_added_constraint:
                    independent_constraints.append(constraints.pop())
                    have_added_constraint = True
                    logger.info('Adding this constraint')
                else:
                    constraints_to_retry.append(constraints.pop())
                    logger.info('Retrying this constraint')
        constraints += constraints_to_retry

    return independent_constraints
