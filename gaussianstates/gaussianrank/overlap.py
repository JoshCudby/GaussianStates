import qutip
import sympy as sp
from gaussianstates.states.gaussian_states import gaussian_states

if __name__ == '__main__':
    m = 50
    n = 8
    dim = 2 ** n
    magic = 1 / 2 * (qutip.basis(dim, 0) + qutip.basis(dim, 15) + qutip.basis(dim, 240) + qutip.basis(dim, 255))

    gaussians = gaussian_states(m, n)
    max_overlap = 0
    max_overlap_state = None
    for i in range(m):
        state = gaussians[:, i]
        overlap = abs(qutip.dag(magic) * state)[0]
        if overlap > max_overlap:
            max_overlap = overlap
            max_overlap_state = state

    print(f'max overlap: {max_overlap}')
    x = sp.symbols('x')
    required_a = sp.solve(1 / 3 * (-x + (x**2 + 3)**0.5) - max_overlap, x)
    print(f'required A: {required_a}')
