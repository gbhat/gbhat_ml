import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian


def f(x):
    return x ** 2


def minimize_sgd(x):
    tfx = tf.Variable(x)
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    for i in range(0, 1000):
        opt.minimize(lambda: f(tfx), var_list=[tfx])
    return np.round(tfx.numpy(), 3), np.round(f(tfx).numpy(), 3)


def fun_jac(x):
    return Jacobian(lambda x: f(x))(x).ravel()


def minimize_lbfgs(x):
    res = minimize(f, x, method='L-BFGS-B', jac=fun_jac)
    return np.round(res.x[0], 3), np.round(f(res.x[0]), 3)


min_x, min_y = minimize_sgd(8.0)
print("Gradient descent minimum value: (", min_x, ",", min_y, ")")


min_x, min_y = minimize_lbfgs(8.0)
print("L-BFGS-B minimum value: (", min_x, ",", min_y, ")")
