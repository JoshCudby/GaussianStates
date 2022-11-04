import sys
import os
print(os.getcwd())
sys.path.extend([os.getcwd()])
print(sys.path)

from gaussianstates.constraints import gaussian_constraints as constraints
from gaussianstates.utils import logging_utils, file_reading_utils
import time

# ~~~~~~~~~~~~ Parameters controlling script ~~~~~~~~~~~~~~
dim = 6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    logger = logging_utils.get_formatted_logger(__name__)
    start_time = time.time()
    logger.info(f'Resuming run for n = {dim}')

    independent_constraints = constraints.resume_partial_run_independent_mp(dim)
    logger.info(f'Execution time: {round(time.time() - start_time, 2)}')

    if independent_constraints is not None:
        logger.info(f'Saving independent constraints for n = {dim}')
        filename = f'data/MpRecursiveConstraints/independent_constraints_{dim}.npy'
        file_reading_utils.save_list_np_array(independent_constraints, filename)

    logger.info('Finished script')
