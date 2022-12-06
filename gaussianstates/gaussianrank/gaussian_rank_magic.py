import qutip
import numpy as np
from typing import List
from scipy.optimize import minimize, NonlinearConstraint

from gaussianstates.constraints.independent_constraints_direct import get_small_set_targets
from gaussianstates.states.gaussian_states import gaussian_states
from gaussianstates.utils import logging_utils
from gaussianstates.utils.binary_string_utils import strings_with_weight, read_binary_array
from gaussianstates.utils.constraint_utils import get_constraints_from_targets

"""Decomposes tensor products of a magic state into a sum of Gaussian states"""
logger = logging_utils.get_formatted_logger(__name__)


def _magic_state() -> qutip.Qobj:
    return 1 / (2 ** 0.5) * (qutip.basis(16, 0) + qutip.basis(16, 15))


def grad_cost_function(state_labels: List[List[complex]]):
    penalty = 3

    even_weight_bin = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
        for item in sublist
    ]
    even_weight_labels = [read_binary_array(b) for b in even_weight_bin]

    special_labels = [0, 15, 240, 255]
    grad = np.zeros((3, len(even_weight_labels) + 1), dtype=complex)
    for j, state_label in enumerate(state_labels):
        if not len(state_label) == len(even_weight_labels) + 1:
            logger.error(state_label)
            logger.error(len(state_label))
            logger.error(len(even_weight_labels))
            raise Exception("State label should be a const and then the amplitudes.")

        for label_index, label in enumerate(even_weight_labels):
            grad[j][label_index] = (
                -0.5 * state_label[-1] if label in special_labels else 0
                + penalty * (
                    sum([
                        state_label[even_weight_labels.index(
                            term[(i + 1) % 2])] * (
                            -1) ** term_index
                        for c in constraints for
                        term_index, term in enumerate(c) for
                        i, t in enumerate(term)
                        if term[i] == label
                    ])
                    + abs(state_label[label_index])
                )
            )

        grad[j][-1] = (
            -0.5 * (
                state_label[even_weight_labels.index(0)] + state_label[even_weight_labels.index(15)] +
                state_label[even_weight_labels.index(240)] + state_label[even_weight_labels.index(255)]
            )
        )
    return grad


def cost_function(state_labels):
    return abs(1 - 1 / 4 * (
        sum([
            state_labels[x + 256] * (sum([state_labels[x + 2 * even_weight_labels.index(special_label)]]))
            - state_labels[x + 257] * (sum([state_labels[x + 2 * even_weight_labels.index(special_label) + 1]]))
            for special_label in special_labels for x in [0, 258, 516]
        ]) ** 2
        +
        sum([
            state_labels[x + 256] * (sum([state_labels[x + 2 * even_weight_labels.index(special_label) + 1]]))
            + state_labels[x + 257] * (sum([state_labels[x + 2 * even_weight_labels.index(special_label)]]))
            for special_label in special_labels for x in [0, 258, 516]
        ]) ** 2
    ))


def normalisation_constraints(state_labels):
    return [np.sum(state_labels[x: x + 256] ** 2) for x in [0, 258, 516]]


def gaussian_constraints(state_labels):
    g_constraints = []
    for x in [0, 258, 516]:
        for c in constraints:
            # real part
            g_constraints.append(
                sum([
                    (
                        (state_labels[x + 2 * even_weight_labels.index(t[0])]
                         * state_labels[x + 2 * even_weight_labels.index(t[1])])
                        - (state_labels[x + 2 * even_weight_labels.index(t[0]) + 1]
                           * state_labels[x + 2 * even_weight_labels.index(t[1]) + 1])
                    ) * (-1) ** index
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
                    ) * (-1) ** index
                    for index, t in enumerate(c)
                ])
            )
    return g_constraints


def all_constraints(state_labels):
    return [*gaussian_constraints(state_labels), *normalisation_constraints(state_labels)]


def _find_gaussian_rank_magic():
    """Decomposes tensor products of a magic state into a sum of Gaussian states"""
    start_states = gaussian_states(3, n)
    start_weight = 1 / (3 ** 0.5)

    def map_state(state, weight):
        return np.array(
            [[np.real(t), np.imag(t)] for t in [*state, weight] if not t == 0], dtype='float64'
        ).reshape(258)

    initial_random_states = np.array([
        map_state(start_states[:, 0], start_weight),
        map_state(start_states[:, 1], start_weight),
        map_state(start_states[:, 2], start_weight)
    ], dtype='float64').reshape(258 * 3)

    nonlinear_constraint = NonlinearConstraint(
        all_constraints,
        [0] * 3 * 2 * len(constraints) + [1] * 3,
        [0] * 3 * 2 * len(constraints) + [1] * 3
    )
    logger.info('Starting minimize')
    logger.info(cost_function(initial_random_states))
    solution = minimize(
        cost_function,
        initial_random_states,
        method='trust-constr',
        options={'verbose': 3},
        constraints=nonlinear_constraint
    )
    return solution


def main():
    s = _find_gaussian_rank_magic()
    return s


n = 8
constraints = get_constraints_from_targets(get_small_set_targets(n))

even_weight_bin = [
    item for sublist in
    [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
    for item in sublist
]
even_weight_labels = [read_binary_array(b) for b in even_weight_bin]

special_labels = [0, 15, 240, 255]

if __name__ == '__main__':
    sol = main()
