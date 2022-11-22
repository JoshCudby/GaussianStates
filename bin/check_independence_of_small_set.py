import sys
import os
import time
import numpy as np

sys.path.extend([os.getcwd()])
from gaussianstates.constraints import independent_constraints_direct as constraints
from gaussianstates.utils import logging_utils
from scipy.spatial.distance import hamming
from gaussianstates.utils.binary_string_utils import strings_with_weight

# ~~~~~~~~~~~~ Parameters controlling script ~~~~~~~~~~~~~~
dim = 8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    logger = logging_utils.get_formatted_logger(__name__)
    logger.info(f'Computing constraints for n = {dim} ...')

    start_time = time.time()
    low_rank = []
    high_rank = []
    for i in range(1):
        independent_constraints, J, x = constraints.get_independent_constraints_directly_from_small_set(dim)
        if np.linalg.matrix_rank(J) < len(independent_constraints):
            low_rank.append(sorted(x))
        else:
            high_rank.append(sorted(x))

    # target_set = [
    #     # [np.array([0, 0, 1, 1, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([0, 1, 0, 1, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([0, 1, 1, 0, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([0, 1, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 0, 0, 1, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 0, 1, 0, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 0, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 1, 0, 0, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 1, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 1, 0, 1, 1, 1]), np.array([0, 0, 0, 0, 0, 1])],
    #     # [np.array([1, 1, 1, 1, 1, 0]), np.array([0, 0, 0, 0, 0, 1])],
    #     # #
    #     # [np.array([0, 1, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 1, 0])],
    #     # [np.array([1, 0, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 1, 0])],
    #     # [np.array([1, 1, 0, 1, 0, 0]), np.array([0, 0, 0, 0, 1, 0])],
    #     # [np.array([1, 1, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 1, 0])],
    #     # #
    #     # [np.array([1, 1, 1, 0, 1, 1]), np.array([0, 0, 0, 1, 0, 0])],
    #     # #
    #     # [np.array([0, 1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 0, 1])],
    #     # [np.array([0, 1, 1, 1, 0, 0]), np.array([1, 1, 0, 0, 0, 1])],
    #     # [np.array([0, 1, 1, 0, 1, 0]), np.array([1, 1, 0, 0, 0, 1])],
    #     # [np.array([0, 1, 1, 1, 1, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     # [np.array([0, 0, 0, 0, 1, 0]), np.array([1, 0, 1, 0, 0, 1])],
    #     # [np.array([0, 0, 0, 1, 0, 0]), np.array([1, 1, 0, 0, 0, 1])],
    #     # [np.array([0, 0, 0, 1, 1, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     # [np.array([0, 0, 1, 0, 0, 0]), np.array([1, 1, 0, 0, 0, 1])],
    #     # [np.array([0, 0, 1, 0, 1, 1]), np.array([1, 0, 1, 1, 0, 0])],
    #     # [np.array([0, 0, 1, 1, 0, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     # [np.array([0, 0, 1, 1, 1, 0]), np.array([1, 1, 0, 0, 0, 1])],
    #     # #
    #     # [np.array([1, 0, 1, 1, 1, 1]), np.array([0, 1, 0, 0, 0, 0])],
    #     # [np.array([1, 0, 1, 1, 0, 0]), np.array([0, 1, 0, 0, 1, 1])],
    #     # [np.array([1, 0, 1, 0, 1, 0]), np.array([0, 1, 0, 1, 0, 1])],
    #     # [np.array([1, 0, 0, 1, 1, 0]), np.array([0, 1, 0, 0, 0, 0])],
    #     # #
    #     # [np.array([1, 1, 1, 1, 1, 0]), np.array([0, 0, 0, 1, 1, 1])],
    #     # #
    #     [np.array([1, 0, 1, 1, 1, 1]), np.array([0, 1, 0, 0, 0, 0])],
    #     [np.array([0, 0, 1, 0, 1, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 0, 1, 1, 0, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 0, 1, 1, 1, 0]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 0, 0, 1, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 0, 1, 1, 0]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 1, 0, 0, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 1, 0, 1, 0]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0, 0])],
    #     [np.array([0, 1, 1, 1, 1, 1]), np.array([1, 0, 0, 0, 0, 0])],
    #     #
    #     [np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 0, 0, 0, 0])],
    #     [np.array([0, 0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 0, 0])],
    #     [np.array([0, 0, 1, 0, 1, 1]), np.array([0, 1, 0, 0, 0, 0])],
    #     [np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 0, 0, 0, 0])],
    #     #
    #     [np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 1, 0, 0, 0])],
    #     #
    # ]
    # independent_constraints, x = constraints.get_independent_constraints_directly_from_target_set(
    #     target_set,
    #     6
    # )
    # t0 = [t[0] for t in target_set]
    # odd_weight = [
    #     item for sublist in
    #     [strings_with_weight(6, k) for k in range(1, 7, 2)]
    #     for item in sublist
    # ]
    # distances = []
    # for e in odd_weight:
    #     count = 0
    #     for arr in t0:
    #         if hamming(e, arr) * 6 > 2:
    #             count += 1
    #     distances.append(count)

    logger.info('Computed constraints')
    logger.info(f'Execution time: {round(time.time() - start_time, 2)}')
