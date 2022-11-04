import sys
import os

from gaussianstates.constraints import gaussian_constraints as constraints
from gaussianstates.states import gaussian_states as states
from gaussianstates.utils import logging_utils, file_reading_utils
import time

# ~~~~~~~~~~~~ Parameters controlling script ~~~~~~~~~~~~~~
dim = 6
should_overwrite = True
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    x = dim
    independent_constraints = []

    logger = logging_utils.get_formatted_logger(__name__)
    logger.info('Starting script')

    directory_name = 'data/MpRecursiveConstraints'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    filename = directory_name + f'/independent_constraints_%s.npy'

    while x > 3:
        try:
            if should_overwrite:
                raise FileNotFoundError
            independent_constraints = file_reading_utils.load_list_np_array(filename % x)
            logger.info(f'Loaded constraints for n = {x}')
            break
        except FileNotFoundError:
            logger.info(f'No constraints found for n = {x}')
            x -= 2

    x += 2

    while x < dim + 1:
        start_time = time.time()
        logger.info(f'Computing constraints for n = {x}')
        state = states.gaussian_states(1, x)
        independent_constraints = constraints.get_independent_constraints_for_next_order_mp(
            independent_constraints,
            x,
            filename % x
        )
        constraints.verify_constraints(independent_constraints, state)
        logger.info(f'Execution time: {round(time.time() - start_time, 2)}')

        logger.info(f'Saving independent constraints for n = {x}')
        file_reading_utils.save_list_np_array(independent_constraints, filename % x)
        x += 2

    logger.info('Finished script')
