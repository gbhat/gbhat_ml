import tensorflow as tf
import numpy as np


optimizers = [
    tf.keras.optimizers.SGD(learning_rate=0.1),
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.96),
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.96, nesterov=True),
]

optimizer_names = ["Gradient Descent", "Gradient Descent with Momentum", "Gradient Descent with Nesterov Momentum"]


def f(x):
    return x ** 2 * tf.math.sin(x)


for i, opt in enumerate(optimizers):
    tfx = tf.Variable(-4.87)
    for x in range(0, 1000):
        val = opt.minimize(lambda: f(tfx), var_list=[tfx])
    print("Algorithm: ", optimizer_names[i], ", Minimum value: (", np.round(tfx.numpy(), 3), ",", np.round(f(tfx).numpy(), 3), ")")
