from Code.Constraints.GaussianConstraints import *
from Code.States.GaussianStates import gaussian_states
from Code.Utils.FileReading import *
from Code.Utils.Logging import get_formatted_logger
from time import time
import os

logger = get_formatted_logger(__name__)
logger.info('Starting script')

dim = 6

directory_name = f'../Output/RecursiveConstraints'
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
filename = directory_name + f'/independent_constraints_%s.npy'

x = dim

independent_constraints = []
while x > 3:
    try:
        independent_constraints = load_list_np_array(filename % x)
        logger.info(f'Loaded constraints for n = {x}')
        break
    except FileNotFoundError:
        logger.info(f'No constraints found for n = {x}')
        x -= 2

x += 2
while x < dim + 1:
    start_time = time()
    logger.info(f'Computing constraints for n = {x}')
    state = gaussian_states(1, x)
    independent_constraints = get_independent_constraints_for_next_order(independent_constraints, x, filename % x)
    verify_constraints(independent_constraints, state)
    logger.info(f'Execution time: {round(time() - start_time, 2)}')

    logger.info(f'Saving independent constraints for n = {x}')
    save_list_np_array(independent_constraints, filename % x)
    x += 2

logger.info('Finished script')
