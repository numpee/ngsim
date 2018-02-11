import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import math

# Units are in FEET!

filepath = 'trajectories-0400-0415.csv'
data = pd.read_csv(filepath)
data_cut = data[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y','Lane_ID','Angle', 'v_Length', 'v_Width']]
sorted_frame = data_cut.sort_values(by=['Frame_ID'])
sorted_np = sorted_frame.values
sorted_np = sorted_np[1000:30000,:]

# init array of sliced values, by frame number
sliced = []

# slice data by frame number
for i in range(int(min(sorted_np[:,1])),int(max(sorted_np[:,1]))):
    sliced.append(sorted_np[sorted_np[:,1]==i])

#fig, ax = plt.subplots()
img = plt.imread("ASPeachtree.jpg")
fig = plt.figure(figsize=(18,6))
#ax = fig.add_axes([0,0,1,1],frameon=False)
# ax = fig.add_subplot(2,1,2)
ax1 = fig.add_subplot(1,1,1)
#fig, ax = plt.subplots()

# patch = patches.Rectangle((0, 0), 0, 0, fc='y', angle=30)


def animate(i):
    x = sliced[i][:,2]
    y = sliced[i][:,3]
    angles = sliced[i][:,5]
    names = sliced[i][:,0]
    lane_label = sliced[i][:,4]
    vehicle_length = sliced[i][:,6]
    vehicle_width = sliced[i][:,7]

    # ax.clear()
    ax1.clear()
    #ax.imshow(img, extent = [-300,300,0,1500])
    # ax.set_autoscaley_on(False)
    # ax.set_autoscalex_on(False)
    # ax.set_xlim([200,330])
    # ax.set_ylim([0,100])
    # ax.scatter(y,x, s = 10)

    ax1.set_autoscaley_on(False)
    ax1.set_autoscalex_on(False)
    ax1.set_xlim([200,400])
    ax1.set_ylim([0,100])
    # ax1.scatter(y,x,s=10)
    patches = []
    patches1 = []
    lane_color = ["white", "red", "orange", "yellow", "green", "blue", "black", "pink"]
    # ax1.scatter(y,x, s = 50, marker = "s")
    for x_cent, y_cent,car_angles, lane, vlength, vwidth in zip(x,y,angles,lane_label,vehicle_length, vehicle_width):
        print(x_cent, y_cent, car_angles)
        if car_angles<25 and car_angles>-25:
            patches.append(ax1.add_patch(plt.Rectangle((y_cent-vlength/2, x_cent-vwidth/2), vlength, vwidth,
                        fill=False, angle=car_angles, linewidth = 2, edgecolor = lane_color[int(lane)])))
        else:
            patches.append(ax1.add_patch(plt.Rectangle((y_cent-vlength/2, x_cent-vwidth/2), vlength, vwidth,
                        fill=False, angle=0, linewidth = 2, edgecolor = lane_color[int(lane)])))

        #patches1.append(ax.add_patch(plt.Rectangle((y_cent, x_cent), 3, 2, fill=False, edgecolor="blue",label=lane_label)))
    # for i, txt in enumerate(names):
    #      ax1.annotate(int(txt), (int(y[i]),int(x[i])), fontsize=10)
    # if i%2==0:
    #     return patches
    # else:
    #     return patches1
    return patches


# transcribed at 100ms intervalS
ani = animation.FuncAnimation(fig, animate, frames = range(2,30000), interval=100, blit=True)
plt.show()
