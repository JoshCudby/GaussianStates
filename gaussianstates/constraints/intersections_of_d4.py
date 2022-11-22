import random
from scipy.spatial.distance import hamming
from gaussianstates.utils.binary_string_utils import strings_with_weight
from gaussianstates.utils.logging_utils import get_formatted_logger
import numpy as np
from typing import List

logger = get_formatted_logger(__name__)


def find_intersections_of_d4(strings: List[np.ndarray]) -> List[np.ndarray]:
    return [
        e for e in even_weight if all([hamming(e, s) * 6 > 2 for s in strings])
    ]


if __name__ == '__main__':
    n = 6
    even_weight = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
        for item in sublist
    ]
    string_list = [even_weight[0]] + even_weight[16:19]  # + [even_weight[21]]
    distances = [
        hamming(string_list[i], string_list[j]) * n
        for i in range(len(string_list) - 1)
        for j in range(i + 1, len(string_list))
    ]
    intersection = find_intersections_of_d4(string_list)
    logger.info(len(intersection))


