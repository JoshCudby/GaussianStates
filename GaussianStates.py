import functools
import itertools
from typing import List, Any

import numpy as np
import qutip


def jordan_wigner(n, j):
    # n is the number of qubits, j is the operator number
    k = int(np.ceil(j/2))
    operator_components = []
    for z in range(k - 1):
        operator_components.append(qutip.sigmaz())
    operator_components.append(qutip.sigmay() if j % 2 == 0 else qutip.sigmax())
    for i in range(n-k):
        operator_components.append(qutip.identity(2))
    return functools.reduce(qutip.tensor, operator_components)


def gaussian_states(m, n):
    # m is the number of states desired, output as columns in a matrix; n is number of qubits
    size = 2 * n
    jordan_wigners = [jordan_wigner(n, i + 1) for i in range(size)]
    states = []
    for i in range(m):
        random_components = (np.random.random_sample((size, size)) - 0.5) * 2
        hamiltonian = qutip.qzero([n, n])
        for k, l in itertools.product(range(size), range(size)):
            hamiltonian = hamiltonian + random_components[k][l] * jordan_wigners[k] * jordan_wigners[l]
            operator = (1j * hamiltonian).expm()
        states.append(operator.full()[:, 0])
    return states


print(gaussian_states(3, 2))
