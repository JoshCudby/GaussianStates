from Code.Constraints.GaussianConstraints2 import get_all_constraints

# Would be more efficient to just run for a single large i, printing/saving at each step if desired
for i in range(4, 7, 1):
    c = get_all_constraints(i)
    for cons in c:
        print(list(map(list, cons)))
    print(len(c))
