import numpy as np
import pandas as pd
import math

filepath = 'trajectories-0400-0415.csv'
data = pd.read_csv(filepath)

# only uptil car #996
data = data[['Vehicle_ID', 'Frame_ID', 'Local_X',
        'Local_Y','v_Vel', 'Lane_ID']]
data_np = data.values.astype(np.float32)
#num_data = 30000
num_data = len(data_np)
data_np = data_np[0:num_data]

sliced = []
#slice data by vehicle ID
for i in range(int(min(data_np[:,0])),int(max(data_np[:,0]))+1):
    sliced.append(data_np[data_np[:,0]==i])

dx, dy, theta, x_p, y_p, v, lane_num, d0 = ([] for i in range(8))

for z in range(0,len(sliced)):
#i = 0       #for debugging with one set only
    x_pos = sliced[z][:,2]
    y_pos = sliced[z][:,3]
    vel = sliced[z][:,4]
    lane = sliced[z][:,5]

    for count in range(len(x_pos)):
        d0.append(x_pos[count]%12)      #calculate distance between from lane
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
    # add to total data set. Technically, can be done by extracting from data set, but
    # there may be indexing issues (number of data points)
    x_p = np.append(x_p, x_pos)
    y_p = np.append(y_p, y_pos)
    v = np.append(v, vel)
    lane_num = np.append(lane_num, lane)


# now concatenate

features = pd.DataFrame({'x_position': [], 'y_position': [], 'theta': [], 'lane': []})
features = features.assign(vehicle_id = data['Vehicle_ID'], frame = data_np[:,1],
                           x_position = x_p, y_position = y_p, velocity = v, lane = lane_num,
                           theta = theta, d0=d0)

'''Now we must iterate through the whole list to find d0~d6 and v0~v6. Could be more efficient to
sort by frame first, but then it becomes harder to add back into data '''

d1, d2, d3, d4, d5, d6, v1,v2,v3,v4,v5,v6 = ([] for i in range(12))
for i in range(0,num_data):
    # init values behind as zero, infront as 1000
    ve2, ve4, ve6 = (0 for f in range(3))
    di1, di2, di3, di4, di5, di6, ve1, ve3, ve5 = (1000 for n in range(9))
    #obtain relevant data
    vehicle_x = x_p[i]
    vehicle_y = y_p[i]
    frame = data_np[i,1]
    lane = lane_num[i]

    # extract all cars infront and behind current car at that frame
    infront = features.loc[(features['frame']==frame) & (features['y_position']>vehicle_y),
                           ['x_position','y_position', 'velocity', 'lane']]
    behind = features.loc[(features['frame']==frame) & (features['y_position']<vehicle_y),
                          ['x_position','y_position', 'velocity', 'lane']]
    # for each lane above and below
    infront_same = infront.loc[(infront['lane']==lane)]
    behind_same = behind.loc[(behind['lane']==lane)]
    infront_higher = infront.loc[(infront['lane']==lane+1)]
    behind_higher = behind.loc[(behind['lane']==lane+1)]
    infront_lower = infront.loc[(infront['lane']==lane-1)]
    behind_lower = behind.loc[(behind['lane']==lane-1)]

    # extract features now
    if not infront_same.empty:
        index1 = infront_same['y_position'].idxmin()
        di1 = features['y_position'][index1]-vehicle_y
        ve1 = features['velocity'][index1]
    if not behind_same.empty:
        index2 = behind_same['y_position'].idxmax()
        di2 = vehicle_y-features['y_position'][index2]
        ve2 = features['velocity'][index2]
    if not infront_higher.empty:
        index5 = infront_higher['y_position'].idxmin()
        di5 = abs(features['y_position'][index5]-vehicle_y)
        ve5 = features['velocity'][index5]
    if not behind_higher.empty:
        index6 = behind_higher['y_position'].idxmax()
        di6 = abs(features['y_position'][index6]-vehicle_y)
        ve6 = features['velocity'][index6]
    if not infront_lower.empty:
        index3 = infront_lower['y_position'].idxmin()
        di3 = abs(features['y_position'][index3]-vehicle_y)
        ve3 = features['velocity'][index3]
    if not behind_lower.empty:
        index4= behind_lower['y_position'].idxmax()
        di4 = abs(features['y_position'][index4]-vehicle_y)
        ve4 = features['velocity'][index4]
    d1.append(di1); d2.append(di2); d3.append(di3); d4.append(di4); d5.append(di5); d6.append(di6);
    v1.append(ve1); v2.append(ve2); v3.append(ve3); v4.append(ve4); v5.append(ve5); v6.append(ve6);

    #track progress
    if i%100 ==0:
        print('Currently at i = {}'.format(i))

features= features.assign(d1 = d1, d2=d2,d3=d3,d4=d4,d5=d5,d6=d6,v1=v1,v2=v2,v3=v3,v4=v4,v5=v5,v6=v6)
features.to_csv('features_redone.csv')
