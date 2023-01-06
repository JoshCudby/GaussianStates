import random
from scipy.spatial.distance import hamming
from gaussianstates.utils.binary_string_utils import strings_with_weight, read_binary_array, int_to_binary_array
from gaussianstates.utils.constraint_utils import get_targets, get_constraints_from_targets, make_jacobian
from gaussianstates.utils.logging_utils import get_formatted_logger
import numpy as np
from typing import List

logger = get_formatted_logger(__name__)


def independent_constraints_for_labels(labels: List[int], n: int) -> List[np.ndarray]:
    targets = get_targets(n)
    constraints = get_constraints_from_targets(targets)
    test_constraints = random.sample(constraints, len(labels))
    J = make_jacobian(test_constraints, labels, n)
    count = 0
    max_count = 15
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


def count_all_but_3_bit_agreements(labels: List[int]):
    labels_in_bin = [int_to_binary_array(v, n) for v in labels]
    count = 0
    for i1 in range(len(labels)):
        for i2 in range(i1 + 1, len(labels)):
            for i3 in range(i2 + 1, len(labels)):
                for i4 in range(i3 + 1, len(labels)):
                    test_labels = [labels_in_bin[i1], labels_in_bin[i2], labels_in_bin[i3], labels_in_bin[i4]]
                    agree_vec = [
                        1 if all([label[i] == 0 for label in test_labels]) or
                             all([label[i] == 1 for label in test_labels]) else 0
                        for i in range(len(test_labels[0]))
                    ]
                    if hamming(agree_vec, [0] * n) * n > n - 4:
                        count += 1
                        logger.info(
                            [labels_in_bin[i1][i] if agree_vec[i] == 1 else None for i in range(len(agree_vec))]
                        )
    return count


def count_all_but_4_bit_agreements(labels: List[int]):
    labels_in_bin = [int_to_binary_array(v, n) for v in labels]
    counts = {}
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                iterator = 0
                for l in labels_in_bin:
                    if l[i1] == b[0] and l[i2] == b[1]:
                        iterator += 1
                counts[(i1, i2, b[0], b[1])] = iterator
    return counts


if __name__ == '__main__':
    n = 8
    starter = [0] * n
    if not len(starter) == n:
        raise Exception()
    even_weight_bin = [
        item for sublist in
        [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
        for item in sublist
    ]
    even_weight_labels = [read_binary_array(b) for b in even_weight_bin]

    for i in range(0):
        labels = [read_binary_array(e) for e in even_weight_bin if hamming(e, starter) * n > 2]
        original_labels = np.copy(labels)

        for j in range(len(labels)):
            if random.random() > 0.3:
                label_to_change = labels[j]
                label_in_bin = int_to_binary_array(label_to_change, n)
                new_label_in_bin = [0] * len(label_in_bin)
                for k in range(len(label_in_bin)):
                    if label_in_bin[k] == 1 and random.random() > 0.4:
                        new_label_in_bin[k] = 1
                new_label = read_binary_array(new_label_in_bin)
                while not (
                        hamming(new_label_in_bin, [0] * n) * n % 2 == 0 and
                        (new_label == label_to_change or new_label not in labels)
                ):
                    new_label_in_bin = [0] * len(label_in_bin)
                    for k in range(len(label_in_bin)):
                        if label_in_bin[k] == 1 and random.random() > 0.4:
                            new_label_in_bin[k] = 1
                    new_label = read_binary_array(new_label_in_bin)
                labels[j] = new_label

        independent_constraints = independent_constraints_for_labels(labels, n)
        if len(independent_constraints) == 0:
            logger.info('Indep' if len(independent_constraints) > 0 else 'Not indep')
            logger.info(labels)
            size_4_count = count_all_but_3_bit_agreements(sorted(labels))
            size_8_count = count_all_but_4_bit_agreements(labels)
            break


    def sorting_key(constraint):
        return sum(constraint[0])


    constraints = sorted(get_constraints_from_targets(get_targets(n)), key=sorting_key)

    number_labels = 2 ** (n - 1) - int(n * (n - 1) / 2) - 1
    # labels = random.sample(even_weight_labels, number_labels)
    labels = [0, 15, 240, 255]
    J = make_jacobian(constraints, labels, n)
    rank = np.linalg.matrix_rank(J)
    # while rank != number_labels:
    #     logger.info(labels)
    #     labels = random.sample(even_weight_labels, number_labels)
    #     J = make_jacobian(constraints, labels, n)
    #     rank = np.linalg.matrix_rank(J)
    # count = count_all_but_3_bit_agreements(labels)
