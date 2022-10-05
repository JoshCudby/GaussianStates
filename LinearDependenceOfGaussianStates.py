from GaussianStates import *
import sympy


def linear_dependence_of_gaussians(n):
    # n is the number of qubits
    gaussians = gaussian_states(2 ** (n - 1), n)
    reduced_gaussians = sympy.Matrix(gaussians.T).rref()
    return reduced_gaussians


def linear_dependence_of_nn_gaussians(n):
    # n is the number of qubits
    gaussians = nn_gaussian_states(10, n, 0).T

    # Shows that they span the space, but not what states are themselves Gaussian
    reduced_gaussians = sympy.Matrix(gaussians).rref()
    pivots = reduced_gaussians[1]

    # # for n = 4 this works, from the 8 terms like 1000 x 0111
    # constraint_result = [
    #     gaussian[6] * gaussian[9] - gaussian[5] * gaussian[10]
    #     + gaussian[3] * gaussian[12] - gaussian[0] * gaussian[15]
    #     for gaussian in gaussians
    # ]

    # for n = 6 this works for 100000 x 011111
    constraint_1 = [
        gaussian[0] * gaussian[63] - gaussian[48] * gaussian[15]
        + gaussian[40] * gaussian[23] - gaussian[36] * gaussian[27]
        + gaussian[34] * gaussian[29] - gaussian[33] * gaussian[30]
        for gaussian in gaussians
    ]

    # 1110000 x 0001111
    constraint_2 = [
        gaussian[24] * gaussian[39] - gaussian[40] * gaussian[23]
        + gaussian[48] * gaussian[15] - gaussian[60] * gaussian[3]
        + gaussian[58] * gaussian[5] - gaussian[57] * gaussian[6]
        for gaussian in gaussians
    ]

    # 100000 x 101111
    constraint_3 = [
        gaussian[40] * gaussian[39] - gaussian[36] * gaussian[43]
        + gaussian[34] * gaussian[45] - gaussian[33] * gaussian[46]
        for gaussian in gaussians
    ]

    # 110001 x 111110
    constraint_4 = [
        gaussian[57] * gaussian[54] - gaussian[53] * gaussian[58]
        + gaussian[51] * gaussian[60] - gaussian[48] * gaussian[63]
        for gaussian in gaussians
    ]
    return [constraint_1, constraint_2, constraint_3, constraint_4]


dim = 6
res = linear_dependence_of_nn_gaussians(dim)
for r in res:
    print(r)
    print('-----')
