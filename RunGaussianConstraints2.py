from Code.Constraints.GaussianConstraints2 import get_all_constraints
from Code.States.GaussianStates import gaussian_states
from Code.Utils.BinaryStringUtils import strings_with_weight, read_binary_array
import numpy as np
import math

# Would be more efficient to just run for a single large i, printing/saving at each step if desired
for dim in range(7, 8, 1):
    all_constraints = get_all_constraints(dim)
    state = gaussian_states(1, dim)
    for cons in all_constraints:
        # mapped = list(map(list, cons))
        # print(mapped)
        val = 0
        for j in range(len(cons)):
            term = cons[j]
            val += ((-1) ** j) * state[term[0]] * state[term[1]]
        if abs(val) > 10 ** (-12):
            print(cons)
            print(val)

    print(len(all_constraints))

    symbolic_vars = []
    for num in range(2, 1001):  # should scale with dim
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            sqrt = math.sqrt(num)
            while sqrt > 10:
                sqrt /= 10
            symbolic_vars.append(sqrt)

    indep_constraints = []
    test_constraints = []
    for z in range(len(all_constraints)):
        test_constraints = list(np.copy(indep_constraints))
        test_constraints.append(all_constraints[z])
        M = len(test_constraints)
        x_values_binary: list[np.ndarray] = [
            item for sublist in [strings_with_weight(dim, k) for k in range(2, dim + 1, 2)] for item in sublist
        ][0:M - 1] + [np.zeros(dim)]
        x_values = [read_binary_array(bin_arr) for bin_arr in x_values_binary]
        J = np.zeros((M, M))
        count = 0
        labels_added = [-1] * len(symbolic_vars)
        for k in range(M):
            constraint = test_constraints[k]
            for l in range(M):
                x = x_values[l]
                for constraint_index in range(len(constraint)):
                    constraint_term = constraint[constraint_index]
                    for index in range(2):
                        if x == constraint_term[index]:
                            label_to_add = constraint_term[(index + 1) % 2]
                            if label_to_add not in labels_added:
                                # Maintain a list of the labels we have already added
                                labels_added[count] = label_to_add
                                count += 1
                            for symbols_index in range(len(labels_added)):
                                if label_to_add == labels_added[symbols_index]:
                                    J[k, l] = symbolic_vars[symbols_index] * ((-1) ** constraint_index)

        # floating point errors
        det = np.linalg.det(J)
        if det > 10 ** (-12):
            indep_constraints.append(all_constraints[z])
    # for cons in indep_constraints:
    #     mapped = list(map(list, cons))
    #     print(mapped)
    print(len(indep_constraints))




# Very dirty and inefficient code to use the IFT
    # indep_constraints = []
    # test_constraints = []
    # for z in range(len(all_constraints)):
    #     test_constraints = list(np.copy(indep_constraints))
    #     test_constraints.append(all_constraints[z])  # det becomes zero when we include the 5th constraint
    #     M = len(test_constraints)
    #     x_values_binary: list[np.ndarray] = strings_with_weight(i, 2)
    #     x_values_binary = x_values_binary[0:M]
    #     x_values = [read_binary_array(bin_arr) for bin_arr in x_values_binary]
    #     J = zeros(M, M)
    #     a, b, c, d, e, f, g, h, m, n, o, p, q, r, s, t, u, v, w = symbols("a b c d e f g h m n o p q r s t u v w")
    #     symbolic_vars = [a, b, c, d, e, f, g, h, m, n, o, p, q, r, s, t, u, v, w]
    #     count = 0
    #     int_vars = [-1] * len(symbolic_vars)
    #     for k in range(M):
    #         constraint = test_constraints[k]
    #         for l in range(M):
    #             x = x_values[l]
    #             for constraint_index in range(len(constraint)):
    #                 constraint_term = constraint[constraint_index]
    #                 for index in range(2):
    #                     if x == constraint_term[index]:
    #                         label_to_add = constraint_term[(index + 1) % 2]
    #                         if label_to_add not in int_vars:
    #                             int_vars[count] = label_to_add
    #                             count += 1
    #                         for an_index in range(len(int_vars)):
    #                             if label_to_add == int_vars[an_index]:
    #                                 J[k, l] = symbolic_vars[an_index] * ((-1) ** constraint_index)
    #
    #     det = J.det()
    #     if det != 0:
    #         indep_constraints.append(all_constraints[z])
    #         print(det)
    #         print(J)
    # print(indep_constraints)