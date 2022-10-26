import numpy as np
from typing import List
from .Logging import get_formatted_logger

logger = get_formatted_logger(__name__)


def verify_constraints(constraints: List[np.ndarray], state: np.ndarray) -> None:
    for cons in constraints:
        val = 0
        for j in range(len(cons)):
            term = cons[j]
            val += ((-1) ** j) * state[term[0]] * state[term[1]]
        if abs(val) > 10 ** (-12):
            logger.error(cons)
            logger.error(val)
            raise Exception('Constraint not satisfied')
