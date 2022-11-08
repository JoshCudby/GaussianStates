import sys
import os
sys.path.extend([os.getcwd()])

from gaussianstates.constraints import independent_constraints_direct as constraints
from gaussianstates.states import gaussian_states as states
from gaussianstates.utils import logging_utils, file_reading_utils
import time

logger = logging_utils.get_formatted_logger(__name__)
should_overwrite = True

for dim in range(6, 7, 2):
    state = states.gaussian_states(1, dim)
    filename = f'data/IndependentConstraints/independent_constraints_{dim}.npy'
    try:
        if should_overwrite:
            raise FileNotFoundError
        independent_constraints = file_reading_utils.load_list_np_array(filename)
        logger.info(f'Loaded existing constraints for n = {dim}')
    except FileNotFoundError:
        logger.info(f'Computing constraints for n = {dim} ...')
        start_time = time.time()
        independent_constraints = constraints.get_independent_constraints_directly(dim)
        logger.info('Computed constraints')
        logger.info(f'Execution time: {round(time.time() - start_time, 2)}')
