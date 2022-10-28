#!/usr/bin/env python
import os
from time import time
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool


def matrix_operation(x: int):
    a = np.random.random([2000, 2000])
    a = a + a.T
    b = np.linalg.det(a)
    return b


def matrix_operation_2(x: int):
    a = np.random.random([2000, 2000])
    a = a + a.T
    b = np.linalg.matrix_rank(a)
    return b


if __name__ == '__main__':
    print('Using %d processors' % int(os.getenv('SLURM_CPUS_PER_TASK', 1)))
    print('Using %d threads' % int(os.getenv('OMP_NUM_THREADS', 1)))
    print('Using %d tasks' % int(os.getenv('SLURM_NTASKS', 1)))

    t_start = time()

    with Pool(2) as p:
        res = p.apipe(matrix_operation, 1)
        res2 = p.apipe(matrix_operation_2, 2)
        print(res.get())
        print(res2.get())

    t_delta = time() - t_start

    print('Seconds taken to operate on %d symmetric 2000x2000 matrices: %f' % (2, t_delta))
