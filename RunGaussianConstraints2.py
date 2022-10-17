from Code.Constraints.GaussianConstraints2 import get_all_constraints, find_independent_constraints
from Code.States.GaussianStates import gaussian_states

# Would be more efficient to just run for a single large i, printing/saving at each step if desired
for dim in range(10, 11, 2):
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
    independent_constraints = find_independent_constraints(all_constraints, state)
    long_constraints = [constraint for constraint in independent_constraints if len(constraint) == dim]
    print(f'{len(long_constraints)} highest order constraints')
    # flattened_constraints = [constraint.flatten() for constraint in long_constraints]
    # a_labels = [item for sublist in flattened_constraints for item in sublist]
    # a_labels_set = sorted(list(set(a_labels)))
    # for a in a_labels_set:
    #     count = 0
    #     for a_test in a_labels:
    #         if a_test == a:
    #             count += 1
    #     print(f'{a}: {count}')

