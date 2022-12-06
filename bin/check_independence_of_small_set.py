import sys
import os
import time
import numpy as np

sys.path.extend([os.getcwd()])
from gaussianstates.constraints import independent_constraints_direct as constraints
from gaussianstates.utils import logging_utils

# ~~~~~~~~~~~~ Parameters controlling script ~~~~~~~~~~~~~~
dim = 8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    logger = logging_utils.get_formatted_logger(__name__)
    logger.info(f'Computing constraints for n = {dim} ...')

    start_time = time.time()
    low_rank = []
    high_rank = []
    independent_constraints = []
    for i in range(1):
        independent_constraints, J, x = constraints.get_independent_constraints_directly_from_small_set(dim)
        if np.linalg.matrix_rank(J) < len(independent_constraints):
            low_rank.append(sorted(x))
        else:
            high_rank.append(sorted(x))

    logger.info('Computed constraints')
    logger.info(f'Execution time: {round(time.time() - start_time, 2)}')

    def sorting_key(constraint):
        return sum(constraint[0])

    independent_constraints = sorted(independent_constraints, key=sorting_key)
