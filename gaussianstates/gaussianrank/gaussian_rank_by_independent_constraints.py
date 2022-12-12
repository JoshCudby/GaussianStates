import scipy.spatial.distance as distance
import sympy as sp
from gaussianstates.constraints.independent_constraints_direct import get_small_set_targets
from gaussianstates.gaussianrank.gaussian_rank_magic import s_key
from gaussianstates.utils.binary_string_utils import strings_with_weight, read_binary_array, int_to_binary_array
from gaussianstates.utils.constraint_utils import get_constraints_from_targets, remove_duplicates
from gaussianstates.utils.logging_utils import get_formatted_logger

logger = get_formatted_logger(__name__)

chi = 3
n = 8
unsorted_constraints = get_constraints_from_targets(get_small_set_targets(n))
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

weight_2_bin = strings_with_weight(n, 2)
weight_2_labels = [0] + sorted([read_binary_array(b) for b in weight_2_bin])

special_labels = [0, 15, 240, 255]
special_indices = [even_weight_labels.index(x) for x in special_labels]


def get_a_val(a_label: int, indep_vars: sp.Matrix):
    index_in_bin = int_to_binary_array(a_label, n)
    if distance.hamming(index_in_bin, [0] * n) * n < 4:
        return indep_vars[weight_2_labels.index(a_label)]

    relevant_constraint = [c for c in constraints if any([t == a_label for term in c for t in term])][0]
    val = 0
    for i, term in enumerate(relevant_constraint[1:len(relevant_constraint)]):
        val += ((-1) ** i) \
               * get_a_val(term[0], indep_vars) \
               * get_a_val(term[1], indep_vars)
    val /= indep_vars[even_weight_labels.index(relevant_constraint[0][0])]
    return val


def main():
    a_indep_vars = sp.symarray('a', (chi, int(n * (n - 1) / 2) + 1))
    c_vars = sp.symbols('c0:%d' % chi)
    all_indep_vars = [a for sublist in a_indep_vars for a in sublist] + [c for c in c_vars]

    sum_equations = []
    for i in range(2 ** (n - 1)):
        sum_equations.append(
            sum([
                c_vars[j] * get_a_val(even_weight_labels[i], a_indep_vars[j][:]) for j in range(chi)
            ])
            - (sp.Rational(1, 2) if even_weight_labels[i] in special_indices else 0)
        )

    normalisation_equations = []
    for j in range(chi):
        normalisation_equations.append(sum([
            abs(get_a_val(even_weight_labels[i], a_indep_vars[j][:])) ** 2 for i in range(2 ** (n - 1))
        ]) - 1)

    logger.info('Starting solve')
    system = sum_equations + normalisation_equations
    solution = sp.nonlinsolve(system, all_indep_vars)
    return solution


if __name__ == '__main__':
    s = main()
