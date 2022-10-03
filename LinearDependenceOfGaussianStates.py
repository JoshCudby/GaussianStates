from GaussianStates import *
import sympy


def pretty_print(v):
    out = []
    for i in v:
        out.append(np.around(i, 2))
    print(out)


def linear_dependence_of_gaussians(n):
    # n is the number of qubits
    gaussians = gaussian_states(2 ** (n - 1), n)
    for gaussian in gaussians.T:
        print(gaussian)
    reduced_gaussians = sympy.Matrix(gaussians.T).rref()
    return reduced_gaussians


def linear_dependence_of_nn_gaussians(n):
    # n is the number of qubits
    gaussians = np.concatenate(
        (nn_gaussian_states(2 ** (n-1), n, 0), nn_gaussian_states(2**(n-1), n, 1)),
        1
    )
    for gaussian in gaussians.T:
        pretty_print(gaussian)
    reduced_gaussians = sympy.Matrix(gaussians.T).rref()
    return reduced_gaussians


dim = 4
print(linear_dependence_of_nn_gaussians(dim))
