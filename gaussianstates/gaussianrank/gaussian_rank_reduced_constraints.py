import numpy as np
import sympy as sp
from gaussianstates.constraints.independent_constraints_direct import get_targets
from gaussianstates.gaussianrank.gaussian_rank_magic import s_key
from gaussianstates.utils.binary_string_utils import strings_with_weight, read_binary_array
from gaussianstates.utils.constraint_utils import get_constraints_from_targets, remove_duplicates
from gaussianstates.utils.logging_utils import get_formatted_logger

logger = get_formatted_logger(__name__)

chi = 3
n = 8
unsorted_constraints = get_constraints_from_targets(get_targets(n))
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
special_indices = [even_weight_labels.index(x) for x in special_labels]


def main():
    a_vars = sp.symarray('a', (chi, 3))
    # c_vars = sp.symbols('c0:%d' % chi)
    all_vars = [a for sublist in a_vars for a in sublist]  # + [c for c in c_vars]

    sum_equations = []
    for i in range(3):
        sum_equations.append(
            sum([
                # c_vars[j] * a_vars[j][i] for j in range(chi)
                a_vars[j][i] for j in range(chi)
            ])
            - (sp.Rational(1, 2) if i in special_indices else 0)
        )
    sum_equations.append(
        sum([
            # c_vars[j] * a_vars[j][1] * a_vars[j][2] / a_vars[j][0] for j in range(chi)
            a_vars[j][1] * a_vars[j][2] / a_vars[j][0] for j in range(chi)
        ])
        - sp.Rational(1, 2)
    )

    normalisation_equations = []
    for i in range(chi):
        normalisation_equations.append(
            sum([
                a_vars[i][j] * np.conj(a_vars[i][j]) for j in range(3)
            ])
            + (a_vars[i][1] * a_vars[i][2] / a_vars[i][0]) * np.conj(a_vars[i][1] * a_vars[i][2] / a_vars[i][0])
        )

    logger.info('Starting solve')
    system = sum_equations + normalisation_equations
    solution = sp.solve(system, all_vars)
    return solution


if __name__ == '__main__':
    s = main()
