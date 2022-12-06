import sys
import os
import time
import pickle

sys.path.extend([os.getcwd()])
from gaussianstates.gaussianrank import gaussian_rank_magic
from gaussianstates.utils import logging_utils


if __name__ == '__main__':
    logger = logging_utils.get_formatted_logger(__name__)
    logger.info(f'Computing gaussian rank ...')

    start_time = time.time()

    solution = gaussian_rank_magic.main()

    logger.info('Computed rank')
    logger.info('Solution')
    logger.info(solution.x)
    logger.info('Optimality')
    logger.info(solution.optimality)
    logger.info('Constraint violation')
    logger.info(solution.constr_violation)
    logger.info(f'Execution time: {round(time.time() - start_time, 2)}')

    with open('data/GaussianRank/gaussian_rank_n_8', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(solution, f, pickle.HIGHEST_PROTOCOL)
