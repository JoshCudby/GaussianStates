from Code.Constraints.GaussianConstraints import *
from Code.States.GaussianStates import gaussian_states
from Code.Utils.FileReading import *
from datetime import datetime

for dim in range(12, 13, 2):
    state = gaussian_states(1, dim)
    filename = f'./Output/Constraints/all_constraints_{dim}.npy'
    try:
        all_constraints = load_list_np_array(filename)
    except FileNotFoundError:
        all_constraints = get_all_constraints(dim)
        verify_constraints(all_constraints, state)
        save_list_np_array(all_constraints, filename)

    # print(f'dim = {dim}')
    # print(f'No. constraints = {len(all_constraints)}')

    for i in range(1):
        print(f'Starting run {i}')
        long_constraints = [x for x in all_constraints if len(x) == dim]
        random.shuffle(long_constraints)
        all_constraints[0: len(long_constraints)] = long_constraints
        all_constraints = all_constraints
        print('Getting independent constraints')
        independent_constraints = get_independent_constraints(all_constraints, state)
        now = datetime.now().strftime('%d-%m-%Y-%Hh%Mm%Ss')
        filename = f'./Output/Constraints/independent_constraints_{dim}_{now}.npy'
        print('Saving to file')
        save_list_np_array(independent_constraints, filename)

    # matrix = get_matrix_of_independent_constraint_possibilities(all_constraints, 300, dim)
    # for r in matrix:
    #     print(r)
    # print(np.linalg.matrix_rank(matrix))

    # seen_constraints = get_constraints_seen_for_targets(
    #     all_constraints,
    #     indexes=[0, 2, 4, 7],
    #     state=state,
    #     dim=dim,
    #     number_of_runs=800
    # )

d = datetime.now().strftime('%d-%m-%Y-%Hh%Mm%Ss')
