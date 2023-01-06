import qutip
import numpy as np
from typing import List
from scipy.optimize import minimize, NonlinearConstraint, basinhopping

from gaussianstates.constraints.independent_constraints_direct import get_small_set_targets
from gaussianstates.states.gaussian_states import gaussian_states
from gaussianstates.utils import logging_utils
from gaussianstates.utils.binary_string_utils import strings_with_weight, read_binary_array
from gaussianstates.utils.constraint_utils import get_constraints_from_targets, verify_constraints, remove_duplicates

"""Decomposes tensor products of a magic state into a sum of Gaussian states"""
logger = logging_utils.get_formatted_logger(__name__)


def _magic_state() -> qutip.Qobj:
    return 1 / (2 ** 0.5) * (qutip.basis(16, 0) + qutip.basis(16, 15))


def real_part_sum(state_labels: np.ndarray):
    return sum([
        state_labels[x + 2 ** n]
        * (sum([state_labels[x + 2 * even_weight_labels.index(special_label)]]))
        - state_labels[x + 2 ** n + 1]
        * (sum([state_labels[x + 2 * even_weight_labels.index(special_label) + 1]]))
        for special_label in special_labels for x in np.linspace(0, (chi - 1) * (2 ** n + 2), chi, dtype='int')
    ])


def imag_part_sum(state_labels: np.ndarray):
    return sum([
        state_labels[x + 2 ** n]
        * (sum([state_labels[x + 2 * even_weight_labels.index(special_label) + 1]]))
        + state_labels[x + 2 ** n + 1]
        * (sum([state_labels[x + 2 * even_weight_labels.index(special_label)]]))
        for special_label in special_labels for x in np.linspace(0, (chi - 1) * (2 ** n + 2), chi, dtype='int')
    ])


def cost_function(state_labels):
    return abs(
        1 - 1 / 4 * (real_part_sum(state_labels) ** 2 + imag_part_sum(state_labels) ** 2)
    )


def grad_cost_function(state_labels):
    grad = np.zeros(chi * (2 ** n + 2))
    for x in np.linspace(0, (chi - 1) * (2 ** n + 2), chi, dtype='int'):
        for special_label in special_labels:
            grad[x + 2 * even_weight_labels.index(special_label)] = - 1 / 2 * (
                state_labels[x + 2 ** n] * real_part_sum(state_labels)
                + state_labels[x + 2 ** n + 1] * imag_part_sum(state_labels)
            )
            grad[x + 2 * even_weight_labels.index(special_label) + 1] = - 1 / 2 * (
                - state_labels[x + 2 ** n + 1] * real_part_sum(state_labels)
                + state_labels[x + 2 ** n] * imag_part_sum(state_labels)
            )

        grad[x + 2 ** n] = - 1 / 2 * (
            sum([
                state_labels[x + 2 * even_weight_labels.index(special_label)] for special_label in special_labels
            ]) * real_part_sum(state_labels)
            + sum([
                state_labels[x + 2 * even_weight_labels.index(special_label) + 1] for special_label in special_labels
            ]) * imag_part_sum(state_labels)
        )
        grad[x + 2 ** n + 1] = - 1 / 2 * (
            - sum([
                state_labels[x + 2 * even_weight_labels.index(special_label) + 1] for special_label in special_labels
            ]) * real_part_sum(state_labels)
            + sum([
                state_labels[x + 2 * even_weight_labels.index(special_label)] for special_label in special_labels
            ]) * imag_part_sum(state_labels)
        )

    return grad


def normalisation_constraints(state_labels):
    return [np.sum(state_labels[x: x + 2 ** n] ** 2) for x in
            np.linspace(0, (chi - 1) * (2 ** n + 2), chi, dtype='int')]


def gaussian_constraints(state_labels):
    g_constraints = []

    for x in np.linspace(0, (chi - 1) * (2 ** n + 2), chi, dtype='int'):
        for c in constraints:
            # real part
            g_constraints.append(
                sum([
                    (
                        (state_labels[x + 2 * even_weight_labels.index(t[0])]
                         * state_labels[x + 2 * even_weight_labels.index(t[1])])
                        - (state_labels[x + 2 * even_weight_labels.index(t[0]) + 1]
                           * state_labels[x + 2 * even_weight_labels.index(t[1]) + 1])
                    ) * ((-1) ** index)
                    for index, t in enumerate(c)
                ])
            )
            # imaginary part
            g_constraints.append(
                sum([
                    (
                        (state_labels[x + 2 * even_weight_labels.index(t[0])]
                         * state_labels[x + 2 * even_weight_labels.index(t[1]) + 1])
                        + (state_labels[x + 2 * even_weight_labels.index(t[0]) + 1]
                           * state_labels[x + 2 * even_weight_labels.index(t[1])])
                    ) * ((-1) ** index)
                    for index, t in enumerate(c)
                ])
            )
    return g_constraints


def all_constraints(state_labels):
    return [*gaussian_constraints(state_labels), *normalisation_constraints(state_labels)]


def _find_gaussian_rank_magic():
    """Decomposes tensor products of a magic state into a sum of Gaussian states"""
    start_states = gaussian_states(chi, n)
    start_weight = 1 / (chi ** 0.5)

    def map_state(state, weight):
        return np.array(
            [[np.real(t), np.imag(t)] for t in [*state, weight] if not t == 0], dtype='float64'
        ).reshape(2 ** n + 2)

    initial_random_states = np.array([
        map_state(start_states[:, i], start_weight) for i in range(chi)
    ], dtype='float64').reshape((2 ** n + 2) * chi)

    nonlinear_constraint = NonlinearConstraint(
        all_constraints,
        [0] * chi * 2 * len(constraints) + [1] * chi,
        [0] * chi * 2 * len(constraints) + [1] * chi
    )
    logger.info('Starting minimize')
    logger.info(cost_function(initial_random_states))
    verify_constraints(constraints, start_states[:, 0])

    solution = minimize(
        cost_function,
        initial_random_states,
        method='SLSQP',
        # jac=grad_cost_function,
        options={'verbose': 3},
        constraints=nonlinear_constraint,
    )
    return solution


def main():
    s = _find_gaussian_rank_magic()
    return s


def s_key(c):
    return sum(c[0])


chi = 1
n = 8
unsorted_constraints = get_constraints_from_targets(get_small_set_targets(n)) \
                       + get_constraints_from_targets(get_small_set_targets(n, [0, 0, 0, 0, 1, 1, 1, 1])) \
                       + get_constraints_from_targets(get_small_set_targets(n, [1, 1, 1, 1, 0, 0, 0, 0])) \
                       + get_constraints_from_targets(get_small_set_targets(n, [1, 1, 1, 1, 1, 1, 1, 1]))
constraints = sorted(
    remove_duplicates(unsorted_constraints),
    key=s_key
)

even_weight_bin = [
    item for sublist in
    [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
    for item in sublist
]
even_weight_labels = sorted([read_binary_array(b) for b in even_weight_bin])

special_labels = [0, 15, 240, 255]

if __name__ == '__main__':
    sol = main()
