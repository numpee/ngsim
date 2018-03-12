import numpy as np
import pandas as pd
import tensorflow as tf

filepath = "features_redone.csv"
data = pd.read_csv(filepath)

#data already sorted by vehicle id. For batch processing, sort by timestamp?
data = data[['vehicle_id', 'frame', 'velocity', 'theta', 'd0', 'd1',
             'd2', 'd3', 'd4', 'd5', 'd6', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']]
data_np = data.values.astype(np.float32)
data_length = len(data_np)


def remove_unnecessary_data(data, data_length):
    
    #first split velocity, theta data into separate array
    output_data = data_np[:,(2,3)]
    
    #shift output data up by one index
    output_data = np.delete(output_data, (0), axis = 0)
    
    #delete rows of data in which vehicle ID switches
    temp_id = data_np[0,0]
    delete_rows = []
    for i in range(0,data_length):
        if(data_np[i,0] != temp_id):
            temp_id = data_np[i,0]
            delete_rows.append(i-1)
    
    #delete rows of unused data
    data_np_processed = np.delete(data_np, delete_rows, axis=0)
    
    #delete column for input
    data_np_processed = np.delete(data_np_processed, (0,1,2,3), axis=1)
    output_data_processed = np.delete(output_data, delete_rows, axis=0)
    return data_np_processed, output_data_processed
            
    
    
    