import numpy as np
from qutip import *


def random_matchgate():
    random_floats = np.random.default_rng().random(6)
    random_complexes: list[complex] = [complex(random_floats[i], random_floats[i+1]) for i in range(0, 6, 2)]
    random_complexes: list[complex] = [i / i.__abs__() for i in random_complexes]

    a = rand_dm(
        [random_complexes[0], random_complexes[1]]
    )
    b = rand_dm(
        [random_complexes[2], random_complexes[0] * random_complexes[1] / random_complexes[2]]
    )
    print(a)
    print('----')
    print(b)


random_matchgate()
