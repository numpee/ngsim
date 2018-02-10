import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# Units are in FEET!

filepath = 'trajectories-0400-0415.csv'
data = pd.read_csv(filepath)
data_cut = data[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y']]
sorted_frame = data_cut.sort_values(by=['Frame_ID'])
sorted_np = sorted_frame.values
sorted_np = sorted_np[0:30000,:]

# init array of sliced values, by frame number
sliced = []

# slice data by frame number
for i in range(int(min(sorted_np[:,1])),int(max(sorted_np[:,1]))):
    sliced.append(sorted_np[sorted_np[:,1]==i])

#fig, ax = plt.subplots()
img = plt.imread("ASPeachtree.jpg")
fig = plt.figure(figsize=(18,7))
#ax = fig.add_axes([0,0,1,1],frameon=False)
ax = fig.add_subplot(2,1,1)
ax1 = fig.add_subplot(2,1,2)
#fig, ax = plt.subplots()

# patch = patches.Rectangle((0, 0), 0, 0, fc='y', angle=30)


def animate(i):
    x = sliced[i][:,2]
    y = sliced[i][:,3]
    names = sliced[i][:,0]
    ax.clear()
    ax1.clear()
    #ax.imshow(img, extent = [-300,300,0,1500])
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    ax.set_xlim([0,1600])
    ax.set_ylim([0,100])
    ax.scatter(y,x, s = 10)

    ax1.set_autoscaley_on(False)
    ax1.set_autoscalex_on(False)
    ax1.set_xlim([0,600])
    ax1.set_ylim([0,70])
    ax1.scatter(y,x,s=10)
    patches = []
    # ax1.scatter(y,x, s = 50, marker = "s")
    for x_cent, y_cent in zip(x,y):
        print(x_cent, y_cent)
        patches.append(ax1.add_patch(plt.Rectangle((y_cent, x_cent), 4, 2, fill=False, edgecolor="blue")))

    for i, txt in enumerate(names):
         ax1.annotate(int(txt), (int(y[i]),int(x[i])), fontsize=10)
    return patches

# transcribed at 100ms intervalS
ani = animation.FuncAnimation(fig, animate, frames = range(2,30000), interval=100, blit=True)
plt.show()
