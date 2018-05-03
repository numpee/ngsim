import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import math
import time
import os, sys
from matplotlib.transforms import Affine2D

#COMMENT IF NOT SAVING VIDEO
#ff_path = os.path.join('C:/Program Files/', 'ImageMagick', 'ffmpeg.exe')
#plt.rcParams['animation.ffmpeg_path'] = ff_path
#if ff_path not in sys.path: sys.path.append(ff_path)
#
## This second one will ensure the ".gif" creation works.
#imgk_path = os.path.join('C:/Program Files/', 'ImageMagick', 'convert.exe')
#plt.rcParams['animation.convert_path'] = imgk_path
#if ff_path not in sys.path: sys.path.append(imgk_path)

# Units are in FEET!

print("initializing...")

#Obtain File in CSV, using Pandas
filepath = 'trajectories-0400-0415.csv'
data = pd.read_csv(filepath)

#Keep useful data, sort by Frame ID. Graphing is done by frame
data_cut = data[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y','Lane_ID', 'v_Length', 'v_Width']]

#reduce size
data_cut = data_cut.loc[data_cut['Vehicle_ID']%2!=0]
sorted_frame = data_cut.sort_values(by=['Frame_ID'])
sorted_np = sorted_frame.values
sorted_np = sorted_np[40000:90000,:]         # Omit data upto 100*1000ms = 100s
sorted_id = data_cut.values

# init array of sliced values, by frame number
sliced = []

# slice data by frame number
for i in range(int(min(sorted_np[:,1])),int(max(sorted_np[:,1]))):
    sliced.append(sorted_np[sorted_np[:,1]==i])

def currentLane(y):
    return y//12        # lane has height of 12. int division of 12.

# return lane boundaries of upper, current, and lower lanes w.r.t my car
def laneBoundaries(lane):
    above_upper = (lane+2)*12
    above_lower = (lane+1)*12
    current_lower = (lane)*12
    below_lower = (lane-1)*12
    return above_upper, above_lower, current_lower, below_lower

#LOGIC TO FIND DISTANCES
def findDistances(mycar_x, mycar_y, other_x, other_y, mycar_lane):
    # mycar_x, mycar_y are x y position of my car
    # other_x, other_y is the dataset, which contains x y position of other cars at the same time

    # Find lane boundary values w.r.t the lane of my car. Ex, lane =3, then find y boundaries of lane 4, 3 and 2
    above_upper, above_lower, current_lower, below_lower = laneBoundaries(mycar_lane)
    
    # init all distances to 0
    dist_above_infront = dist_current_infront = dist_below_infront = \
    dist_above_behind = dist_current_behind = dist_below_behind = 0
    
    #find indexes of cars in specific lanes from the dataset. np.where returns index values of the datset
    index_above = np.where((other_y<above_upper)&(other_y>=above_lower))[0]
    index_current = np.where((other_y<above_lower)&(other_y>=current_lower))[0]
    index_below = np.where((other_y<current_lower)&(other_y>=below_lower))[0]
    
    # Using the indexes, create an array of x_values of above, current and below lane
    x_above = other_x[index_above]
    x_current = other_x[index_current]
    x_below = other_x[index_below]
    
    #subract by current x values, then find minimum distance

    # NUMPY MASKING. ONLY LEAVES VALUES THAT MATCH THE CONDITION
    x_above_infront = x_above[x_above>mycar_x]       # Numpy masking 
    if (x_above_infront.size != 0):
        dist_above_infront = np.min(x_above_infront) - mycar_x  # find minimum, then subtract by mycar_x
    x_above_behind = x_above[x_above<mycar_x]       # find behind values
    if (x_above_behind.size != 0):
        dist_above_behind = mycar_x - np.max(x_above_behind)    # find maximum of behind, then distance between mycar and this car
    
    #same as above
    x_current_infront = x_current[x_current>mycar_x]
    if (x_current_infront.size != 0):        
        dist_current_infront = np.min(x_current_infront) - mycar_x
    x_current_behind = x_current[x_current<mycar_x]       # find behind values
    if (x_current_behind.size != 0):
        dist_current_behind = mycar_x - np.max(x_current_behind)
       
    x_below_infront = x_below[x_below>mycar_x]        
    if (x_below_infront.size != 0):
        dist_below_infront = np.min(x_below_infront) - mycar_x
    x_below_behind = x_below[x_below<mycar_x]       # find behind values
    if (x_below_behind.size != 0):
        dist_below_behind = mycar_x - np.max(x_below_behind)
    
    return dist_above_infront, dist_current_infront, dist_below_infront, dist_above_behind, dist_current_behind, dist_below_behind
   
#create and follow waypoint
def followWayPoint(vehicle_x, vehicle_y, target_x, target_y):
    theta = math.atan2(target_y - vehicle_y, target_x - vehicle_x)
    print("Y: {},\t X: {}".format((target_y), (target_x)))
    return math.degrees(theta)

def createWayPoint(vehicle_x, vehicle_y, target_lane, lookahead):
    target_y = target_lane*12+6-2
    target_x = math.sqrt(lookahead**2 - (target_y-vehicle_y)**2) + vehicle_x
    return target_x, target_y    

# LOGIC TO CHANGE LANE
# This function takes as input the distance values of surrounding cars
# outputs velocity and theta values
def changeLane(current_vel, y_pos, x_pos, lane, dist_above_infront,
               dist_current_infront, dist_below_infront, dist_above_behind,
               dist_current_behind, dist_below_behind):
    clearance_front = 40
    clearance_back = 10
    theta = 0
    lookahead = 40
    above_available=False; below_available = False
    # condition for lane change! (Distance infront too small OR larger distances in other lane)
    if((dist_current_infront <clearance_front)):
        if((dist_above_infront > clearance_front) and (dist_above_behind >clearance_back)):
            above_available = True
        if ((dist_below_infront > clearance_front) and (dist_below_behind > clearance_back)):
            below_available = True
        
        if (above_available and below_available):    # if both OK, change to larger clearance lane
            if(dist_above_infront > dist_below_infront):
                target_x, target_y = createWayPoint(x_pos, y_pos, lane+1, lookahead)
                theta = followWayPoint(x_pos, y_pos, target_x, target_y)
            else:
                target_x, target_y = createWayPoint(x_pos, y_pos, lane-1, lookahead)
                theta = followWayPoint(x_pos, y_pos, target_x, target_y)
        elif (above_available and not below_available): # if only one is ok
            target_x, target_y = createWayPoint(x_pos, y_pos, lane+1, lookahead)
            theta = followWayPoint(x_pos, y_pos, target_x, target_y)
        elif (not above_available and below_available):
            target_x, target_y = createWayPoint(x_pos, y_pos, lane-1, lookahead)
            theta = followWayPoint(x_pos, y_pos, target_x, target_y)
        
        #if neither, reduce speed but follow same lane 
        else:
            current_vel = current_vel - 10
            target_x, target_y = createWayPoint(x_pos, y_pos, lane, lookahead)
            theta = followWayPoint(x_pos, y_pos, target_x, target_y)
    # when no lane change is needed, follow center lane
    else:
        current_vel = 50
        target_x, target_y = createWayPoint(x_pos, y_pos, lane, lookahead)
        theta = followWayPoint(x_pos, y_pos, target_x, target_y)
        
    #print(target_x, target_y)
    return current_vel, theta
    
def prediction():
    ''' 
    INPUT TENSORFLOW PREDICTION CODE HERE
    '''
    vel =0
    theta = 0
    return vel, theta


# set figure size
fig = plt.figure(figsize=(27,3))
#ax = fig.add_axes([0,0,1,1],frameon=False)
# ax = fig.add_subplot(2,1,2)

#add subplot
ax1 = fig.add_subplot(1,1,1)
count = 0
myvehicle_x_pos = 220
myvehicle_y_pos = 32
myvehicle_vel = 40
myvehicle_theta = 0
myvehicle_lane = currentLane(myvehicle_y_pos)
# Animation function

print("Initialization Finished!")

def animate(i):
    global count; global myvehicle_x_pos; global myvehicle_vel; global myvehicle_y_pos; global myvehicle_theta
    timestep = 0.1
    global myvehicle_lane
    
    # Slice relevant information by frame number
    x = sliced[i][:,2]
    y = sliced[i][:,3]
    names = sliced[i][:,0]
    lane_label = sliced[i][:,4]
    vehicle_length = sliced[i][:,5]
    vehicle_width = sliced[i][:,6]
    
    myvehicle_lane = currentLane(myvehicle_y_pos)
    dist_above_infront, dist_current_infront, dist_below_infront, dist_above_behind, dist_current_behind, dist_below_behind \
    = findDistances(myvehicle_x_pos, myvehicle_y_pos, y, x, myvehicle_lane)
    
    # UPDATE MYVEHICLE_VEL and MYVEHICLE_THETA using changelane function
    # or when prediction, using prediction() function
    myvehicle_vel, myvehicle_theta = changeLane(myvehicle_vel, myvehicle_y_pos, myvehicle_x_pos,
                                                myvehicle_lane, dist_above_infront,dist_current_infront,
                                                dist_below_infront, dist_above_behind, dist_current_behind,
                                                dist_below_behind)    

    
    # Update position based on velocity
    myvehicle_x_pos = timestep * math.cos(math.radians(myvehicle_theta))*myvehicle_vel + myvehicle_x_pos
    myvehicle_y_pos = timestep * math.sin(math.radians(myvehicle_theta)) * myvehicle_vel + myvehicle_y_pos

    
    # ax.clear()
    ax1.clear()
    x_lines = [0,421]
    y_lines = [73,73]
    x_lines1 = [0,421]
    y_lines1 = [73+21, 73+6.6]
    plt.plot(x_lines, y_lines, '-w', LineWidth=1.5)
    # x_lines1 = [0,834]
    # y_lines1 = [73+38,73]
    plt.plot(x_lines1, y_lines1, '-w', linestyle ='-', LineWidth = 1.5)
    x_lines2 = [421, 421+146]
    y_lines2= [73,73]
    x_lines3=[421,421+146]
    y_lines3=[73+6.6,73]
    plt.plot(x_lines2,y_lines2, '-w', linestyle='--', LineWidth = 1.5)
    plt.plot(x_lines3,y_lines3, '-w', linestyle='--', LineWidth = 1.5)

    x_lines4 = [0, 421, 421+146, 986,1650]
    y_lines4= [73+40, 73+6.6+12,72+12,73, 73]
    plt.plot(x_lines4, y_lines4, '-w', LineWidth = 1.5)

    x_lines5 = [986, 1650]


    plt.axhline(y=0, color='white', linestyle ='-', LineWidth = 1.5)
    plt.axhline(y=12, color='white', linestyle = '--')
    plt.axhline(y=24, color='white', linestyle = '--')
    plt.axhline(y=36, color='white', linestyle = '--')
    plt.axhline(y=48, color='white', linestyle = '--')
    plt.axhline(y=60, color='white', linestyle = '--')
    # plt.axhline(y=72, color='white', linestyle = '--')
    #ax.imshow(img, extent = [-300,300,0,1500])
    # ax.set_autoscaley_on(False)
    # ax.set_autoscalex_on(False)
    # ax.set_xlim([200,330])
    # ax.set_ylim([0,100])
    # ax.scatter(y,x, s = 10)

    # set autoscale off, set x,y axis
    ax1.set_autoscaley_on(True)
    ax1.set_autoscalex_on(True)
    ax1.set_xlim([0,1650])
    ax1.set_ylim([-10,100])
    ax1.set_facecolor('black')
    ax1.fill_between(x_lines, y_lines, y_lines1, color='gray')
    ax1.fill_between(x_lines2, y_lines2,y_lines3, color='gray')
    ax1.fill_between(x_lines4, y_lines4, 100, color='gray')
    ax1.fill_between([0,1650], -10, 0, color='gray')
    rot = Affine2D().rotate_deg(90)



    # ax1.scatter(y,x,s=10)
    patches = []
    patches1 = []
    lane_color = ["white", "red", "orange", "yellow", "green", "blue", "black", "pink"]
    # ax1.scatter(y,x, s = 50, marker = "s")

    # unzip by category, create rectangle for each car by frame
    for x_cent, y_cent, lane, vlength, vwidth in zip(x,y,lane_label,vehicle_length, vehicle_width):
        # print(x_cent, y_cent)
        vlen = vlength*0.75
        vwid = vwidth*0.75
        # colored vehicles
        # patches.append(ax1.add_patch(plt.Rectangle((y_cent-vlen/2, x_cent-vwid/2), vlen, vwid,
        #                 fill=True, angle=0, linewidth = 2, edgecolor = lane_color[int(lane)], color = lane_color[int(lane)])))

        patches.append(ax1.add_patch(plt.Rectangle((y_cent-vlen/2, x_cent-2), vlen, 4,
                        fill=True, angle=0, linewidth = 2, edgecolor = lane_color[int(lane)], color = 'blue', joinstyle = 'round', 
                        capstyle = 'butt')))

    
    patches.append(ax1.add_patch(plt.Rectangle((myvehicle_x_pos-4,myvehicle_y_pos-2), 8, 4, fill=True,
                        angle = myvehicle_theta, color = 'red', joinstyle = 'round' )))
    count = count +1
    #print("lane {}  below distance {}".format(myvehicle_lane, dist_below_infront))

    return patches


# Animate at interval of 100ms
ani = animation.FuncAnimation(fig, animate, frames = range(2,30000), interval=100, blit=True)
#FFwriter = animation.FFMpegWriter()

plt.show()
#ani.save('video.mp4', writer =FFwriter)





