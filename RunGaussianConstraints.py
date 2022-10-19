from Code.Constraints.GaussianConstraints import *
from Code.States.GaussianStates import gaussian_states

for dim in range(6, 7, 2):
    all_constraints = get_all_constraints(dim)
    state = gaussian_states(1, dim)
    verify_constraints(all_constraints, state)

    # print(f'dim = {dim}')
    # print(f'No. constraints = {len(all_constraints)}')

    get_constraints_seen_for_targets(
        all_constraints,
        indexes=[0, 2, 4, 7],
        state=state,
        dim=dim,
        number_of_runs=800
    )
