import functools
import itertools
import numpy as np
from qutip import *


def jordan_wigner(n, j):
    # n is the number of qubits, j is the operator number
    k = int(np.ceil(j/2))
    operator_components = []
    for z in range(k - 1):
        operator_components.append(sigmaz())
    operator_components.append(sigmay() if j % 2 == 0 else sigmax())
    for i in range(n-k):
        operator_components.append(identity(2))
    return functools.reduce(tensor, operator_components)


def gaussian_states(m: int, n: int) -> np.ndarray:
    # m is the number of states desired, output as columns in a matrix; n is number of qubits
    size = 2 * n
    jordan_wigners: list[Qobj] = [jordan_wigner(n, i + 1) for i in range(size)]
    states_list = np.empty((2 ** n, m), complex)
    for i in range(m):
        random_components = (np.random.random_sample((size, size)) - 0.5) * 2
        random_antisymmetric = (random_components - random_components.T) / 2
        hamiltonian: Qobj = qzero(np.full(n, 2).tolist())
        for k, l in itertools.product(range(size), range(size)):
            hamiltonian = hamiltonian + random_antisymmetric[k][l] * jordan_wigners[k] * jordan_wigners[l]
        operator = (-1 * hamiltonian).expm()
        states_list[:, i] = operator.full()[:, 3]
    return states_list


def nn_gaussian_states(m: int, n: int, parity: int) -> np.ndarray:
    # m is the number of states desired, output as columns in a matrix; n is number of qubits
    size = 2 * n
    jordan_wigners: list[Qobj] = [jordan_wigner(n, i + 1) for i in range(size)]
    states_list = np.empty((2 ** n, m), complex)
    for i in range(m):
        random_components = (np.random.random_sample((size, size)) - 0.5) * 2
        random_antisymmetric = (random_components - random_components.T) / 2
        hamiltonian: Qobj = qzero(np.full(n, 2).tolist())
        for k, l in itertools.product(range(n - 1), range(4)):
            j = 2 * k - 1
            hamiltonian = hamiltonian + random_antisymmetric[j][j+l] * jordan_wigners[j] * jordan_wigners[j+l]
        operator = (-1 * hamiltonian).expm()
        states_list[:, i] = operator.full()[:, parity % 2]
    return states_list
