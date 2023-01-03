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

a_val_dicts = [{}, {}, {}]


def update_dicts(val, state_index, a_label):
    a_val_dicts[state_index][a_label] = val
    if state_index == 0:
        a_val_dicts[2][2 ** n - 1 - a_label] = val
    if state_index == 1:
        a_val_dicts[1][2 ** n - 1 - a_label] = val


def get_a_val(a_label: int, indep_vars: sp.Matrix, state_index: int):
    if a_label in a_val_dicts[state_index].keys():
        return a_val_dicts[state_index][a_label]

    if state_index not in range(chi):
        raise Exception('State index should be in range chi')

    if state_index == 2:
        return get_a_val(2 ** n - 1 - a_label, indep_vars, 0)

    if state_index == 1 and a_label > 2 ** (n - 1) - 1:
        return get_a_val(2 ** n - 1 - a_label, indep_vars, 1)

    index_in_bin = int_to_binary_array(a_label, n)
    if distance.hamming(index_in_bin, [0] * n) * n < 4 and (state_index == 0 or state_index == 1):
        val = indep_vars[state_index][weight_2_labels.index(a_label)]
        update_dicts(val, state_index, a_label)
        return val

    relevant_constraint = [c for c in constraints if any([t == a_label for term in c for t in term])][0]
    val = 0
    for i, term in enumerate(relevant_constraint[1:len(relevant_constraint)]):
        val += ((-1) ** i) \
               * get_a_val(term[0], indep_vars, state_index) \
               * get_a_val(term[1], indep_vars, state_index)
    val /= indep_vars[state_index][even_weight_labels.index(relevant_constraint[0][0])]
    update_dicts(val, state_index, a_label)
    return val


def main():
    a_indep_vars = sp.symarray('a', (chi, int(n * (n - 1) / 2) + 1))
    c_vars = sp.symbols('c0:%d' % chi)

    sum_equations = []
    for i in range(2 ** (n - 1)):
        sum_equations.append(
            sum([
                c_vars[j] * get_a_val(even_weight_labels[i], a_indep_vars, j) for j in range(chi)
            ])
            - (sp.Rational(1, 2) if even_weight_labels[i] in special_indices else 0)
        )

    # normalisation_equations = []
    # for j in range(chi):
    #     normalisation_equations.append(sum([
    #         abs(get_a_val(even_weight_labels[i], a_indep_vars, j)) ** 2 for i in range(2 ** (n - 1))
    #     ]) - 1)

    logger.info('Starting solve')
    # system = sum_equations + normalisation_equations
    # solution = sp.solve_poly_system(system, all_indep_vars)
    # return solution

    groebner = sp.polys.polytools.groebner(sum_equations, method='f5b')
    return groebner


if __name__ == '__main__':
    s = main()
