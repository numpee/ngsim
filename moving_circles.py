import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111)

plt.xlim(-100, 100)
plt.ylim(-100, 100)

width = 5
bars = 25

RB = [] # Establish RB as a Python list
for a in range(bars):
    RB.append(patches.Rectangle((a*15-140,-100), width, 200,
          color="blue", alpha=0.50))

def init():
    for a in range(bars):
        ax.add_patch(RB[a])
    return RB

def animate(i):
    for a in range(bars):
        temp = np.array(RB[i].get_xy())
        temp[0] = temp[0] + 3;
        RB[i].set_xy(temp)
    return RB

anim = animation.FuncAnimation(fig, animate,
                           init_func=init,
                           frames=15,
                           interval=20,
                           blit=True)

plt.show()
