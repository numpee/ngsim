import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

x = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
y = [0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
yaw = [0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2, 1.3]
fig = plt.figure()
plt.axis('equal')
plt.grid()
ax = fig.add_subplot(111)
ax.set_xlim(-0, 40)
ax.set_ylim(-10, 10)

patch = patches.Rectangle((0, 0), 0, 0, fc='y', angle=30)

def init():
    ax.add_patch(patch)
    return patch,

def animate(i):
    patch.set_width(1.2)
    patch.set_height(1.0)
    patch.set_xy([x[i], y[i]])
    patch.angle = -np.rad2deg(yaw[i])
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=len(x),
                               interval=100,
                               blit=True)
plt.show()
