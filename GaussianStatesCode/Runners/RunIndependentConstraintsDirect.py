from GaussianStatesCode.Constraints.IndependentConstraintsDirectAlgorithm import *
from GaussianStatesCode.States.GaussianStates import gaussian_states
from GaussianStatesCode.Utils.FileReading import *
from GaussianStatesCode.Utils.Logging import get_formatted_logger
from time import time

logger = get_formatted_logger(__name__)
should_overwrite = True

for dim in range(12, 13, 2):
    state = gaussian_states(1, dim)
    filename = f'../Output/IndependentConstraints/independent_constraints_{dim}.npy'
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
