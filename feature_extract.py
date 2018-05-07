import numpy as np
import pandas as pd
import math

# Read CSV through Pandas
filepath = 'trajectories-0400-0415.csv'
data = pd.read_csv(filepath)

# Leave data with the following headings only
data = data[['Vehicle_ID', 'Frame_ID', 'Local_X',
        'Local_Y','v_Vel', 'Lane_ID']]
data_np = data.values.astype(np.float32)        # convert to numpy

# NGSIM has 1 million data points. Change num_data to another value to reduce 
# # of data points obtained
num_data = len(data_np)
data_np = data_np[0:num_data]

# Create empty list to organize data by Vehicle ID
sliced = []
#slice data by vehicle ID
for i in range(int(min(data_np[:,0])),int(max(data_np[:,0]))+1):
    sliced.append(data_np[data_np[:,0]==i])

# Create empty list for each variable
dx, dy, theta, x_p, y_p, v, lane_num, d0 = ([] for i in range(8))

'''
Iterate for each vehicle ID. Must iterate for each vehicle because of vehicle orientation
which is found by atan2(dx,dy) over 5 time steps.
''' 
for z in range(0,len(sliced)):

    # z contains vehicle number. Thus, sliced[1][:,2] contains x_pos
    # data for 1st vehicle, and so on.
    x_pos = sliced[z][:,2]
    y_pos = sliced[z][:,3]
    vel = sliced[z][:,4]
    lane = sliced[z][:,5]

    # Finding orientation in radians
    for count in range(len(x_pos)):
        if lane[count] == 7:
            d0.append(abs( (18/721) * y_pos + x_pos + 18)/ (np.sqrt((18/721)**2 + (1))))
        else:
            d0.append(x_pos[count]%12)      #calculate distance between from lane
        
        # Due to noise in data, 5 time steps are used. 
        # The delta_x is found by x(t) - x(t-4), same for delta_y
        # Assume orientation of 0 (parallel to lane) for first 4 timesteps of each vehicle
        if count <4:
            dx.append(0)
            dy.append(0)
            theta.append(0)
        else:
            delta_x = x_pos[count]-x_pos[count-4]
            delta_y = y_pos[count]-y_pos[count-4]
            dx.append(delta_x)
            dy.append(delta_y)
            theta.append(math.atan2(delta_x,delta_y)) # OUTPUT: RADIANS!

    # Append back to list.        
    x_p = np.append(x_p, x_pos)
    y_p = np.append(y_p, y_pos)
    v = np.append(v, vel)
    lane_num = np.append(lane_num, lane)


# Now create new dataframe called features, which is fed into feature extractor below for 
# d1~d6, v1~ v6 calculation

features = pd.DataFrame({'x_position': [], 'y_position': [], 'theta': [], 'lane': []})
features = features.assign(vehicle_id = data['Vehicle_ID'], frame = data_np[:,1],
                           x_position = x_p, y_position = y_p, velocity = v, lane = lane_num,
                           theta = theta, d0=d0)

'''Now we must iterate through the whole list to find d0~d6 and v0~v6. Could be more efficient to
sort by frame first, but then it becomes harder to add back into data '''

# create empty lists for variables
d1, d2, d3, d4, d5, d6, v1,v2,v3,v4,v5,v6 = ([] for i in range(12))# iterate through whole list


for i in range(0,num_data):
    # init values behind as zero, infront as 1000
    ve2, ve4, ve6 = (0 for f in range(3))
    di1, di2, di3, di4, di5, di6, ve1, ve3, ve5 = (1000 for n in range(9))

    #obtain relevant data
    vehicle_x = x_p[i]
    vehicle_y = y_p[i]
    frame = data_np[i,1]
    lane = lane_num[i]
    velocity = v[i]

    # The next two lines collect data for surrounding cars at that specific time step.
    # Procedure: 1) List all vehicles in that frame 2) Leave vehicles with greater y value
    # in variable 'infront', and lower y value in variable 'behind'
    infront = features.loc[(features['frame']==frame) & (features['y_position']>vehicle_y),
                           ['x_position','y_position', 'velocity', 'lane']]
    behind = features.loc[(features['frame']==frame) & (features['y_position']<vehicle_y),
                          ['x_position','y_position', 'velocity', 'lane']]


    # From 'infront' and 'behind', find all vehicles in lane +1, lane, and lane-1 lanes.
    infront_same = infront.loc[(infront['lane']==lane)]
    behind_same = behind.loc[(behind['lane']==lane)]
    infront_higher = infront.loc[(infront['lane']==lane+1)]
    behind_higher = behind.loc[(behind['lane']==lane+1)]
    infront_lower = infront.loc[(infront['lane']==lane-1)]
    behind_lower = behind.loc[(behind['lane']==lane-1)]

    # Feature extractor. 'infront_same.empty' returns true if 'infront_same' list is empty 
    # the list 'infront' returns values in increasing y_position. 
    # Thus, .idxmin() is used to find the index of the closest car infront of subject vehicle.
    # .idxmax() is used to find index of closest car behind vehicle.

    if lane ==7:
        if not infront_same.empty:
            index1 = infront_same['y_position'].idxmin()    
            di1 = features['y_position'][index1]-vehicle_y
            ve1 = features['velocity'][index1]
        if not behind_same.empty:
            index2 = behind_same['y_position'].idxmax()
            di2 = vehicle_y-features['y_position'][index2]
            ve2 = features['velocity'][index2]
        di5 = 0
        ve6 = velocity
        di6 = 0
        ve6 = velocity
        if not infront_lower.empty:
            index3 = infront_lower['y_position'].idxmin()
            di3 = abs(features['y_position'][index3]-vehicle_y)
            ve3 = features['velocity'][index3]
        if not behind_lower.empty:
            index4= behind_lower['y_position'].idxmax()
            di4 = abs(features['y_position'][index4]-vehicle_y)
            ve4 = features['velocity'][index4]
    
    elif lane ==1:
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
        di3 = 0
        di4 = 0
        ve3 = velocity
        ve4 = velocity

    elif lane == 6:
        if not infront_same.empty:
            index1 = infront_same['y_position'].idxmin()    
            di1 = features['y_position'][index1]-vehicle_y
            ve1 = features['velocity'][index1]
        if not behind_same.empty:
            index2 = behind_same['y_position'].idxmax()
            di2 = vehicle_y-features['y_position'][index2]
            ve2 = features['velocity'][index2]
        di5 = 0
        di6 = 0
        ve5 = velocity
        ve6 = velocity
        if not infront_lower.empty:
            index3 = infront_lower['y_position'].idxmin()
            di3 = abs(features['y_position'][index3]-vehicle_y)
            ve3 = features['velocity'][index3]
        if not behind_lower.empty:
            index4= behind_lower['y_position'].idxmax()
            di4 = abs(features['y_position'][index4]-vehicle_y)
            ve4 = features['velocity'][index4]
    else:

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

    # append all values to list.
    d1.append(di1); d2.append(di2); d3.append(di3); d4.append(di4); d5.append(di5); d6.append(di6);
    v1.append(ve1); v2.append(ve2); v3.append(ve3); v4.append(ve4); v5.append(ve5); v6.append(ve6);

    # Track progress during run time. 
    if i%100 ==0:
        print('Currently at i = {}'.format(i))

# Export to CSV file. Data can be accessed in pandas using the d1~d6, v1~v6 headers.
features= features.assign(d1 = d1, d2=d2,d3=d3,d4=d4,d5=d5,d6=d6,v1=v1,v2=v2,v3=v3,v4=v4,v5=v5,v6=v6)
features.to_csv('features_redone_lane7.csv')
