import random
from scipy.spatial.distance import hamming
from gaussianstates.utils.binary_string_utils import strings_with_weight, read_binary_array, int_to_binary_array
from gaussianstates.utils.constraint_utils import get_targets, get_constraints_from_targets, make_jacobian, \
    remove_duplicates
from gaussianstates.utils.logging_utils import get_formatted_logger
import numpy as np
from typing import List

logger = get_formatted_logger(__name__)


def independent_constraints_for_labels(labels: List[int], n: int) -> List[np.ndarray]:
    targets = get_targets(n)
    constraints = remove_duplicates(get_constraints_from_targets(targets))
    test_constraints = random.sample(constraints, len(labels))
    J = make_jacobian(test_constraints, labels, n)
    count = 0
    max_count = 5
    while np.linalg.matrix_rank(J) < len(labels) and count < max_count:
        test_constraints = random.sample(constraints, len(labels))
        J = make_jacobian(test_constraints, labels, n)
        count += 1
    if count == max_count:
        return []
    return test_constraints


def pairwise_distances_for_labels(labels: List[int], n: int):
    distances = {}
    labels_in_bin = [int_to_binary_array(label, n) for label in labels]
    for i in range(len(labels_in_bin) - 1):
        for j in range(i + 1, len(labels_in_bin)):
            d = hamming(labels_in_bin[i], labels_in_bin[j]) * n
            if distances.get(d) is None:
                distances.update({d: 1})
            else:
                distances[d] += 1
    return distances


if __name__ == '__main__':
    n = 6
    starter = [1] * n
    if not len(starter) == n:
        raise Exception()
    even_weight = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
        for item in sublist
    ]
    count = 0
    # subs = [read_binary_array(e) for e in even_weight if e[3] == 1 and e[4] == 0]
    for i in range(0):
        labels = [read_binary_array(e) for e in even_weight if hamming(e, starter) * n > 2]
        original_labels = np.copy(labels)

        # for j in range(len(labels)):
        #     if random.random() > 0.3:
        #         label_to_change = labels[j]
        #         label_in_bin = int_to_binary_array(label_to_change, n)
        #         new_label_in_bin = [0] * len(label_in_bin)
        #         for k in range(len(label_in_bin)):
        #             if label_in_bin[k] == 1 and random.random() > 0.4:
        #                 new_label_in_bin[k] = 1
        #         new_label = read_binary_array(new_label_in_bin)
        #         while not (
        #             hamming(new_label_in_bin, [0] * n) * n % 2 == 0 and
        #             (new_label == label_to_change or new_label not in labels)
        #         ):
        #             new_label_in_bin = [0] * len(label_in_bin)
        #             for k in range(len(label_in_bin)):
        #                 if label_in_bin[k] == 1 and random.random() > 0.4:
        #                     new_label_in_bin[k] = 1
        #             new_label = read_binary_array(new_label_in_bin)
        #         labels[j] = new_label

        # labels[0] = 0
        # labels[7] = 9
        # labels[10] = 48
        # labels[12] = 24
        #
        # labels[3] = subs[i]
        # labels = sorted(labels)

        # logger.info(f'Finding Constraints')
        independent_constraints = independent_constraints_for_labels(labels, n)
        if len(independent_constraints) < len(labels):
            logger.info('Not Indep')
            count += 1
            logger.info(labels)
            distances = pairwise_distances_for_labels(labels, n)
            total_distance = 0
            for d, m in distances.items():
                total_distance += d * m
            logger.info(f'Total distance: {total_distance}')
            logger.info(distances)
        # else:
        # logger.info('Indep')

    even_weight_labels = [read_binary_array(e) for e in even_weight]
    sorted_even_weight_labels = sorted(even_weight_labels)
    sorted_even_weight_binary = [int_to_binary_array(e, n) for e in sorted_even_weight_labels]
    starter_labels = [read_binary_array(e) for e in sorted_even_weight_binary if e[0] == 0 and e[1] == 0]
    for i in range(1):
        even_copy = sorted_even_weight_binary.copy()
        labels = starter_labels + [
            read_binary_array(e) for e in random.sample(
                sorted_even_weight_binary[int(n * (n-1) / 2): 2 ** (n - 1)],
                2 ** (n - 1) - int(n * (n - 1) / 2) - 1 - 2 ** (n - 3)
            )
        ]
        constraints = independent_constraints_for_labels(labels, n)
        if len(constraints) > 0:
            print(labels)
