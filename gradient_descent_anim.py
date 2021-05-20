import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x, record=False):
    y = x ** 2
    if record:
        global val_list
        val_list.append((np.round(x.numpy(), 3), np.round(y.numpy(), 3)))
    return y


val_list = []
tfx = tf.Variable(10.0)
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(0, 1000):
    opt.minimize(lambda: f(tfx, record=True), var_list=[tfx])
    if np.round(tfx.numpy(), 3) == 0.0:  # Added this condition to stop animation at the minimum value
        break

fig, ax = plt.subplots()
x = np.linspace(-10, 10)
y = f(x)
plt.plot(x, y)

def draw_step(i, val_list):
    ax.scatter(val_list[i][0], val_list[i][1], s=80, c='red')


ani = FuncAnimation(fig, draw_step, frames=len(val_list), fargs=(val_list,))
plt.show()