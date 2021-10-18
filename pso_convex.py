import numpy as np
import pyswarms as ps


def f(x):
    x_ = x[:, 0]
    y_ = x[:, 1]
    return x_ ** 2 + y_ ** 2


options = {'c1': 0.5, 'c2': 1.2, 'w': 0.9}
bounds = np.array([[-2, -1], [2, 1]])
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=2, bounds=bounds, options=options)
best_cost, best_pos = optimizer.optimize(f, iters=100)
print("Best position: ", best_pos)
