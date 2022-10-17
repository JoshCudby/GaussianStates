from Code.Constraints.GaussianConstraints2 import get_all_constraints
from Code.States.GaussianStates import gaussian_states
import numpy as np

# Would be more efficient to just run for a single large i, printing/saving at each step if desired
for dim in range(7, 8, 1):
    all_constraints = get_all_constraints(dim)
    state = gaussian_states(1, dim)
    for cons in all_constraints:
        val = 0
        for j in range(len(cons)):
            term = cons[j]
            val += ((-1) ** j) * state[term[0]] * state[term[1]]
        if abs(val) > 10 ** (-12):
            print(cons)
            print(val)

    print(f'dim = {dim}')
    print(f'No. constraints = {len(all_constraints)}')

    indep_constraints = []
    test_constraints = []
    for z in range(len(all_constraints)):
        test_constraints = list(indep_constraints.copy())
        test_constraints.append(all_constraints[z])
        M = len(test_constraints)
        flattened_constraints = [constraint.flatten() for constraint in test_constraints]
        a_labels = [item for sublist in flattened_constraints for item in sublist]
        a_set = sorted(list(set(a_labels)))
        x_values = a_set[0:M]
        J = np.zeros((M, M), dtype=complex)
        count = 0
        for k in range(M):
            constraint = test_constraints[k]
            for l in range(M):
                x = x_values[l]
                for constraint_index in range(len(constraint)):
                    constraint_term = constraint[constraint_index]
                    for index in range(2):
                        if x == constraint_term[index]:
                            label_to_add = constraint_term[(index + 1) % 2]
                            J[k, l] = complex(state[label_to_add]) * ((-1) ** constraint_index)

        rank = np.linalg.matrix_rank(J)
        if rank == M:
            indep_constraints.append(all_constraints[z])
        if z % 1000 == 0:
            print(f'Constraint number {z} reached')
    # for cons in indep_constraints:
    #     mapped = list(map(list, cons))
    #     print(mapped)
    print(f'Number of indep constraints = {len(indep_constraints)}')
