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
sorted_np = sorted_np[0:3000,:]

# init array of sliced values, by frame number
sliced = []

# slice data by frame number
for i in range(int(min(sorted_np[:,1])),int(max(sorted_np[:,1]))):
    sliced.append(sorted_np[sorted_np[:,1]==i])

#fig, ax = plt.subplots()

fig = plt.figure(figsize=(7,7))
ax = fig.add_axes([0,0,1,1],frameon=False)


x = sliced[10][:,2]
y = sliced[10][:,3]
names = sliced[5][:,0]
ax.scatter(x,y)

for i, txt in enumerate(names):
    ax.annotate(txt, (x[i],y[i]))

plt.show()