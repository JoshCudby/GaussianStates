import sympy
from qutip import *
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product
import numpy as np


def matchgate(vector: np.ndarray):
    # Defines a random MG given parameters, using local/non-local decomposition
    mat = tensor(
        (1j * vector[2] * sigmaz()).expm(),
        (1j * vector[3] * sigmaz()).expm()
    ) * (
        1j * (vector[4] * tensor(sigmax(), sigmax()) + vector[5] * tensor(sigmay(), sigmay()))
    ).expm() * tensor(
        (1j * vector[0] * sigmaz()).expm(), (1j * vector[1] * sigmaz()).expm()
    )
    return Qobj(mat, dims=[[2, 2], [2, 2]])


def gaussian_state(c_width, c_size):
    qubit_lines = np.random.default_rng().integers(c_width - 1, size=c_size)
    qc = QubitCircuit(N=c_width)

    for gate_index in range(c_size):
        targets = [qubit_lines[gate_index], qubit_lines[gate_index] + 1]
        random_coefficients = 2 * np.pi * np.random.rand(6)
        qc.user_gates[gate_index] = matchgate(random_coefficients)
        qc.add_gate(gate_index, targets=targets)

    u_list = qc.propagators()
    overall_circuit = gate_sequence_product(u_list)
    basis_ket = ket([0] * c_width, 2)
    return overall_circuit * basis_ket


print(gaussian_state(4, 50))
gaussians = np.zeros((16, 16), dtype=complex)
for i in range(16):
    gaussians[:, [i]] = gaussian_state(4, 50)
reduced_gaussians = sympy.Matrix(gaussians.T).rref()

print(reduced_gaussians)
