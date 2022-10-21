from Code.Constraints.GaussianConstraints import *
from Code.States.GaussianStates import gaussian_states
from Code.Utils.FileReading import *
from Code.Utils.Logging import get_formatted_logger
from datetime import datetime
from time import time
import os

logger = get_formatted_logger(__name__)

for dim in range(10, 13, 2):
    state = gaussian_states(1, dim)
    filename = f'.\\Output\\Constraints\\all_constraints_{dim}.npy'
    try:
        all_constraints = load_list_np_array(filename)
        logger.info(f'Loaded existing constraints for n = {dim}')
    except FileNotFoundError:
        logger.info('No existing constraints found. Computing ...')
        start_time = time()
        all_constraints = get_all_constraints(dim)
        verify_constraints(all_constraints, state)
        logger.info('Computed constraints')
        logger.info(f'Execution time: {round(time() - start_time, 2)}')
        save_list_np_array(all_constraints, filename)

    # logger.info(f'dim = {dim}')
    # logger.info(f'No. constraints = {len(all_constraints)}')

    for i in range(1):
        logger.info(f'Starting independent constraints run number {i + 1}')
        start_time = time()
        long_constraints = [x for x in all_constraints if len(x) == dim]
        random.shuffle(long_constraints)
        all_constraints[0: len(long_constraints)] = long_constraints
        independent_constraints = get_independent_constraints(all_constraints, state)
        logger.info(f'Execution time: {round(time() - start_time, 2)}')

        now = datetime.now().strftime('%d-%m-%Y-%Hh%Mm%Ss')
        directory_name = f'.\\Output\\Constraints\\independent_constraints_{dim}'

        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

        filename = directory_name + f'\\{now}.npy'
        logger.info('Saving to file')
        save_list_np_array(independent_constraints, filename)

    # matrix = get_matrix_of_independent_constraint_possibilities(all_constraints, 300, dim)
    # for r in matrix:
    #     logger.info(r)
    # logger.info(np.linalg.matrix_rank(matrix))

    # seen_constraints = get_constraints_seen_for_targets(
    #     all_constraints,
    #     indexes=[0, 2, 4, 7],
    #     state=state,
    #     dim=dim,
    #     number_of_runs=800
    # )

logger.info('Finished script')
