import tensorflow as tf
import numpy as np


def f(x):
    return x ** 2 * tf.math.sin(x)


optimizers = [
    tf.keras.optimizers.SGD(learning_rate=0.01),
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.95),
    tf.keras.optimizers.RMSprop(learning_rate=0.1),
    tf.keras.optimizers.RMSprop(learning_rate=0.1, momentum=0.95),
]

for i, opt in enumerate(optimizers):
    tfx = tf.Variable(-4.8)
    for x in range(0, 1000):
        val = opt.minimize(lambda: f(tfx), var_list=[tfx])
    config = opt.get_config()
    print("Algorithm: ", config['name'], "(learning rate:",  config['learning_rate'], ", momentum:", config['momentum'],
            " Minimum value: (", np.round(tfx.numpy(), 3), ",", np.round(f(tfx).numpy(), 3), ")")
