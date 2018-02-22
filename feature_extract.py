import numpy as np
import pandas as pd
import math

filepath = 'trajectories-0400-0415.csv'
data = pd.read_csv(filepath)

# only uptil car #996
data = data[['Vehicle_ID', 'Frame_ID', 'Local_X',
        'Local_Y','v_Vel', 'Lane_ID']]
data_np = data.values.astype(np.float32)
data_np = data_np[0:321836]

sliced = []
#slice data by vehicle ID
for i in range(int(min(data_np[:,0])),int(max(data_np[:,0]))+1):
    sliced.append(data_np[data_np[:,0]==i])

dx = []; dy = []; theta = []
for z in range(0,len(sliced)):
#i = 0       #for debugging with one set only
    x_pos = sliced[z][:,2]
    y_pos = sliced[z][:,3]
    vel = sliced[z][:,4]
    lane = sliced[z][:,5]

    for count in range(len(x_pos)):
        if count <4:
            dx.append(0)
            dy.append(0)
            theta.append(0)
        else:
            delta_x = x_pos[count]-x_pos[count-4]
            delta_y = y_pos[count]-y_pos[count-4]
            dx.append(delta_x)
            dy.append(delta_y)
            theta.append(math.atan2(delta_x,delta_y)) #in radians, y=inf x=0 is 0 rad
    
# now concatenate and almost done!