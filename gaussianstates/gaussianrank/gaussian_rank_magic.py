import qutip
import sympy
from gaussianstates.utils import logging_utils

"""Decomposes tensor products of a magic state into a sum of Gaussian states"""
logger = logging_utils.get_formatted_logger(__name__)


def _magic_state() -> qutip.Qobj:
    return 0.5 * (qutip.basis(16, 0) + qutip.basis(16, 5) + qutip.basis(16, 10) + qutip.basis(16, 15))


def _find_gaussian_rank_magic():
    """Decomposes tensor products of a magic state into a sum of Gaussian states"""
    # n = 1
    chi = 2
    number_vars = 16
    # TODO: make this loop terminate with some sensible condition
    while chi < 5:
        logger.info(f'Trying to find a decomposition of rank {chi}')
        # Only need even parity k, not all 16
        x_vars = sympy.Array([list(sympy.symbols('x%d(0:%d)' % (i, number_vars), real=True)) for i in range(chi)])
        y_vars = sympy.Array([list(sympy.symbols('y%d(0:%d)' % (i, number_vars), real=True)) for i in range(chi)])
        a_vars = sympy.Array([[x_vars[i][j] + 1j * y_vars[i][j] for j in range(number_vars)] for i in range(chi)])
        a_star_vars = sympy.Array([[x_vars[i][j] - 1j * y_vars[i][j] for j in range(number_vars)] for i in range(chi)])
        all_vars = [item for sublist in x_vars.tolist() + y_vars.tolist() for item in sublist]

        system = [
            sympy.re(1 / 4 * sum([
                (a_vars[i][k1]) * (a_star_vars[i][k2])
                for i in range(chi) for k1 in range(number_vars) for k2 in range(number_vars)
            ])),
        ]
        for i in range(chi):
            constraint = a_vars[i][0] * a_vars[i][15] + a_vars[i][5] * a_vars[i][10] \
                         - a_vars[i][3] * a_vars[i][12] - a_vars[i][6] * a_vars[i][9]
            new_equations = [
                sympy.re(constraint),
                sympy.im(constraint),
                sum([x_vars[i][k] ** 2 + y_vars[i][k] ** 2 for k in range(number_vars)]) - 1,
                y_vars[i][0]
            ]
            system += new_equations

        solutions = sympy.solve(system, all_vars)
        if len(solutions) > 0:
            logger.info(solutions)
            return system, solutions
        chi += 1


def main():
    system, solutions = _find_gaussian_rank_magic()
    return system, solutions


if __name__ == '__main__':
    S, sols = main()
