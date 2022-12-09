import sys
import os
import time
import pickle

sys.path.extend([os.getcwd()])
from gaussianstates.gaussianrank import gaussian_rank_by_constraints
from gaussianstates.utils import logging_utils


if __name__ == '__main__':
    logger = logging_utils.get_formatted_logger(__name__)
    logger.info(f'Computing gaussian rank ...')
    start_time = time.time()

    solution = gaussian_rank_by_constraints.main()

    logger.info('Computed rank')
    logger.info(solution)
    logger.info(f'Execution time: {round(time.time() - start_time, 2)}')

    filename = f'data/GaussianRank/gaussian_rank_n_8_{round(time.time(), None)}'
    with open(filename, 'wb') as f:
        pickle.dump('solution', f, pickle.HIGHEST_PROTOCOL)
