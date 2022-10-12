from ..Constraints.GaussianConstraints import *

for dim in range(4, 5):
    N = 2 ** dim
    constraints = get_highest_order_constraints(dim, 1)
    print(f'Number of constraints = {np.linalg.matrix_rank(constraints[:, :, 0])}')
    print(constraints[0, :, :])

    if dim < 8:
        _, indep_rows = sympy.Matrix(constraints[:, :, 0]).T.rref()
        unique_constraints = constraints[indep_rows, :, 0]
        verify_highest_order_constraints(dim, unique_constraints)
        print(unique_constraints)

    for number_of_differing_bits in range(4, dim - 1, 2):
        lower_order_constraints = get_lower_order_constraints(dim, number_of_differing_bits)
        print(lower_order_constraints)
# TODO: calculate the lower order constraints
