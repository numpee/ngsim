# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

filepath = 'vehicle-trajectory-data/0400pm-0415pm/trajectories-0400pm-0415pm.csv'
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
fig = plt.figure(figsize=(7,7))
#ax = fig.add_axes([0,0,1,1],frameon=False)
ax = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)
#fig, ax = plt.subplots()


def animate(i):
    x = sliced[i][:,2]
    y = sliced[i][:,3]
    names = sliced[i][:,0]
    ax.clear()
    ax1.clear()
    #ax.imshow(img, extent = [-300,300,0,1500])
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    ax.set_xlim([-300,300])
    ax.set_ylim([0,1500])
    ax.scatter(x,y, s = 10)

    ax1.set_autoscaley_on(False)
    ax1.set_autoscalex_on(False)
    ax1.set_xlim([-100,100])
    ax1.set_ylim([0,600])
    ax1.scatter(x,y, s = 50, marker = "s")

    for i, txt in enumerate(names):
        ax.annotate(int(txt), (int(x[i]),int(y[i])), fontsize=8)
        ax1.annotate(int(txt), (int(x[i]),int(y[i])), fontsize=10)


ani = animation.FuncAnimation(fig,animate,frames = range(2,30000), interval=50)
plt.show()
