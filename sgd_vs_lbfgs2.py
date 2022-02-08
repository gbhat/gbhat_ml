import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


def f(x):
    """ y = e^{-x/2}
        This is the function to be minimized.
        Check the variable type to see if it is called from Tensorflow or Scipy,
        Call the appropriate exp method from tf/np library.
    """
    if isinstance(x, ResourceVariable):
        return tf.math.exp(-x / 2)
    else:
        return np.exp(-x / 2)


def minimize_sgd(x):
    tfx = tf.Variable(x, constraint=lambda z: tf.clip_by_value(z, -10, 0))  # Add boundary constraint (-10, 0)
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    for i in range(0, 1000):
        opt.minimize(lambda: f(tfx), var_list=[tfx])
    return np.round(tfx.numpy(), 3), np.round(f(tfx).numpy(), 3)


def fun_jac(x):
    return Jacobian(lambda x: f(x))(x).ravel()


def minimize_lbfgs(x):
    res = minimize(f, x, method='L-BFGS-B', bounds=((-10, 0),), jac=fun_jac)    # Add bounds to constraint the values in (-10, 0)
    return np.round(res.x[0], 3), np.round(f(res.x[0]), 3)


min_x, min_y = minimize_sgd(9.5)
print("Gradient descent minimum value: (", min_x, ",", min_y, ")")


min_x, min_y = minimize_lbfgs(9.5)
print("L-BFGS-B minimum value: (", min_x, ",", min_y, ")")
