from Code.Constraints.IndependentConstraintsDirectAlgorithm import *
from Code.States.GaussianStates import gaussian_states
from Code.Utils.FileReading import *
from Code.Utils.Logging import get_formatted_logger
from time import time

logger = get_formatted_logger(__name__)
should_overwrite = False

for dim in range(6, 11, 2):
    state = gaussian_states(1, dim)
    filename = f'./Output/IndependentConstraints/independent_constraints_{dim}.npy'
    try:
        if should_overwrite:
            raise FileNotFoundError
        independent_constraints = load_list_np_array(filename)
        logger.info(f'Loaded existing constraints for n = {dim}')
    except FileNotFoundError:
        logger.info(f'Computing constraints for n = {dim} ...')
        start_time = time()
        independent_constraints = get_independent_constraints_directly(dim)
        logger.info('Computed constraints')
        logger.info(f'Execution time: {round(time() - start_time, 2)}')
