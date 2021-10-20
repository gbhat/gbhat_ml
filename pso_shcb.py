import numpy as np
import pyswarms as ps


def f(x):
    x_ = x[:, 0]
    y_ = x[:, 1]
    return ((4 - 2.1 * x_ ** 2 + x_ ** 4 / 3.) * x_ ** 2 + x_ * y_
            + (-4 + 4 * y_ ** 2) * y_ ** 2)


options = {'c1': 0.5, 'c2': 1.2, 'w': 0.9}
bounds = np.array([[-2, -1], [2, 1]])   # Add x limit (-2, 2) and y limit (-1, 1)
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=2, bounds=bounds, options=options)
best_cost, best_pos = optimizer.optimize(f, iters=100)
best_pos = np.round(best_pos, 2)        # Round to 2 decimals
print("Best position: ", best_pos)
