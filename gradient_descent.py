import tensorflow as tf
import numpy as np


def f(x):
    return x ** 2


tfx = tf.Variable(10.0)
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(0, 1000):
    opt.minimize(lambda: f(tfx), var_list=[tfx])

print(" Minimum value: (", np.round(tfx.numpy(), 3), ",", np.round(f(tfx).numpy(), 3), ")")
