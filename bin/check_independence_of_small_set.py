import sys
import os
sys.path.extend([os.getcwd()])

from gaussianstates.constraints import independent_constraints_direct as constraints
from gaussianstates.states import gaussian_states as states
from gaussianstates.utils import logging_utils, file_reading_utils
import time

# ~~~~~~~~~~~~ Parameters controlling script ~~~~~~~~~~~~~~
dim = 14
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    logger = logging_utils.get_formatted_logger(__name__)
    state = states.gaussian_states(1, dim)
    filename = f'data/IndependentConstraints/independent_constraints_small_set_{dim}.npy'

    logger.info(f'Computing constraints for n = {dim} ...')
    start_time = time.time()
    independent_constraints = constraints.get_independent_constraints_directly_from_small_set(dim)
    logger.info('Computed constraints')
    logger.info(f'Execution time: {round(time.time() - start_time, 2)}')
