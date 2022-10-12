from Code.Constraints.GaussianConstraints2 import get_all_constraints, get_highest_order_constraints_even_case
from Code.States.GaussianStates import gaussian_states

# Would be more efficient to just run for a single large i, printing/saving at each step if desired
for i in range(5, 10, 1):
    c = get_all_constraints(i)
    state = gaussian_states(1, i)
    for cons in c:
        mapped = list(map(list, cons))
        val = 0
        for j in range(len(cons)):
            term = cons[j]
            val += ((-1) ** j) * state[term[0]] * state[term[1]]
        if abs(val) > 10 ** (-5):
            print(cons)
            print(val)

    print(len(c))
