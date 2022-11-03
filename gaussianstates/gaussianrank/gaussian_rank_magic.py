import qutip
import sympy
from gaussianstates.utils import logging_utils

logger = logging_utils.get_formatted_logger(__name__)


def magic_state() -> qutip.Qobj:
    return 0.5 * (qutip.basis(16, 0) + qutip.basis(16, 5) + qutip.basis(16, 10) + qutip.basis(16, 15))


def find_gaussian_rank_magic():
    # n = 1
    chi = 2
    # TODO: make this loop terminate with some sensible condition
    while chi < 5:
        logger.info(f'Trying to find a decomposition of rank {chi}')
        x_vars = sympy.Array([list(sympy.symbols('x%d(0:%d)' % (i, 4), real=True)) for i in range(chi)])
        y_vars = sympy.Array([list(sympy.symbols('y%d(0:%d)' % (i, 4), real=True)) for i in range(chi)])
        a_vars = sympy.Array([[x_vars[i][j] + 1j * y_vars[i][j] for j in range(4)] for i in range(chi)])
        a_star_vars = sympy.Array([[x_vars[i][j] - 1j * y_vars[i][j] for j in range(4)] for i in range(chi)])
        all_vars = [item for sublist in x_vars.tolist() + y_vars.tolist() for item in sublist]

        system = [
            1 / 4 * sum([
                (a_vars[i][k1]) * (a_star_vars[i][k2])
                for i in range(chi) for k1 in range(4) for k2 in range(4)
            ]),
        ]
        for i in range(chi):
            new_equations = [
                sympy.re(a_vars[i][0] * a_vars[i][3] + a_vars[i][1] * a_vars[i][2]),
                sympy.im(a_vars[i][0] * a_vars[i][3] + a_vars[i][1] * a_vars[i][2]),
                sum([x_vars[i][k] ** 2 + y_vars[i][k] ** 2 for k in range(4)]) - 1,
                y_vars[i][0]
            ]
            system += new_equations

        solutions = sympy.solve(system, all_vars)
        if len(solutions) > 0:
            logger.info(solutions)
            return system, solutions
        chi += 1


def main():
    lagrangian, derivatives, values, solutions = find_gaussian_rank_magic()
    return lagrangian, derivatives, solutions


if __name__ == '__main__':
    L, D, S = main()
