from qutip import *
import sympy
from Code.Utils.Logging import get_formatted_logger

logger = get_formatted_logger(__name__)


def magic_state() -> Qobj:
    return 0.5 * (basis(16, 0) + basis(16, 5) + basis(16, 10) + basis(16, 15))


def find_gaussian_rank_magic():
    # n = 1
    input = magic_state()
    chi = 2
    logger.info(f'Trying to find a decomposition of rank {chi}')
    while True:
        x_vars = sympy.Array([list(sympy.symbols('x%d(0:%d)' % (i, 4), real=True)) for i in range(chi)])
        y_vars = sympy.Array([list(sympy.symbols('y%d(0:%d)' % (i, 4), real=True)) for i in range(chi)])
        a_vars = sympy.Array([[x_vars[i][j] + 1j * y_vars[i][j] for j in range(4)] for i in range(chi)])
        lambda_vars = sympy.Array([list(sympy.symbols('L%d(0:%d)' % (i, 3), real=True)) for i in range(chi)])
        all_vars = [item for sublist in x_vars.tolist() + y_vars.tolist() + lambda_vars.tolist() for item in sublist]

        lagrangian = 0.25 * sum([
            (a_vars[i][k1]) * (a_vars[i][k2])
            for i in range(chi) for k1 in range(4) for k2 in range(4)
        ])
        lagrangian -= sum([
            lambda_vars[i][0] * (a_vars[i][0] * a_vars[i][3] + a_vars[i][1] * a_vars[i][2])
            + lambda_vars[i][1] * (sum([x_vars[i][k] ** 2 + y_vars[i][k] ** 2 for k in range(4)]) - 1)
            + lambda_vars[i][2] * (y_vars[i][0])
            for i in range(chi)
        ])
        derivatives = [lagrangian.diff(v) for v in all_vars]
        solutions = sympy.nonlinsolve(derivatives, all_vars)
        logger.info(lagrangian)
        logger.info(derivatives)
        logger.info(solutions)
        if chi > 2:
            return False
        chi += 1


def test():
    find_gaussian_rank_magic()


if __name__ == '__main__':
    test()
