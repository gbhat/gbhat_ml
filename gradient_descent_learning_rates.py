import tensorflow as tf
import numpy as np


def f(x):
    global step_count
    step_count += 1
    return x ** 2


optimizers = [
    tf.keras.optimizers.SGD(learning_rate=0.01),
    tf.keras.optimizers.SGD(learning_rate=0.1),
    tf.keras.optimizers.SGD(learning_rate=0.95),
    tf.keras.optimizers.SGD(learning_rate=1.01)
]

step_count = 0
for opt in optimizers:
    tfx = tf.Variable(10.0)
    step_count = 0
    for i in range(0, 1000):
        val = opt.minimize(lambda: f(tfx), var_list=[tfx])
        if np.round(tfx.numpy(), 3) == 0.0:                         # Added this condition to show the difference between learning rates
            break

    print("Learning rate: ", opt.get_config()['learning_rate'], " Minimum value: (", np.round(tfx.numpy(), 3), ",", np.round(f(tfx).numpy(), 3), ") Total steps taken: ", step_count)
