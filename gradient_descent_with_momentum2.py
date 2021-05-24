import tensorflow as tf
import numpy as np


def f(x):
    return x ** 2 * tf.math.sin(x)


sgd_opt = tf.keras.optimizers.SGD(learning_rate=0.01)
sgd_with_momentum_opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.95)

tfx = tf.Variable(-4.87)
for i in range(0, 1000):
    sgd_opt.minimize(lambda: f(tfx), var_list=[tfx])

print(" Minimum value with Gradient Descent : (", np.round(tfx.numpy(), 3), ",", np.round(f(tfx).numpy(), 3), ")")

tfx = tf.Variable(-4.87)
for i in range(0, 1000):
    sgd_with_momentum_opt.minimize(lambda: f(tfx), var_list=[tfx])

print(" Minimum value with Gradient Descent with Momentum : (", np.round(tfx.numpy(), 3), ",", np.round(f(tfx).numpy(), 3), ")")
