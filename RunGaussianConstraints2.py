from Code.Constraints.GaussianConstraints2 import get_all_constraints, find_independent_constraints
from Code.States.GaussianStates import gaussian_states

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
            raise Exception('Constraint not satisfied')

    print(f'dim = {dim}')
    print(f'No. constraints = {len(all_constraints)}')
    find_independent_constraints(all_constraints, state)

