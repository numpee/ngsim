# Import all packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sort_data import remove_unnecessary_data

# import extracted dataset
filepath = "features_redone.csv"
data = pd.read_csv(filepath)

#data already sorted by vehicle id. For batch processing, sort by timestamp?
data = data[['vehicle_id', 'frame', 'velocity', 'theta', 'd0', 'd1',
             'd2', 'd3', 'd4', 'd5', 'd6', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']]
data_np = data.values.astype(np.float32)
len_data = len(data_np)

processed_input, processed_output = remove_unnecessary_data(data_np, len_data)

# Network Parameters
n_hidden1 = 128
n_hidden2 = 128
n_input = 13        # 15 inputs!
n_output = 2        # velocity and theta
batch_size = 128
lr = 0.005

X = tf.placeholder(tf.float32, [None,n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
        'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        'output': tf.Variable(tf.random_normal([n_hidden2, n_output]))        
        }

biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden1])),
        'b2': tf.Variable(tf.random_normal([n_hidden2])),
        'out': tf.Variable(tf.random_normal([n_output]))
        }

def mlp(x):
    #two hidden layer with RELU activation (Use LEAKY RELU?)
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
    
    out_layer = tf.matmul(layer2, weights['output'])+biases['out']
    return out_layer

# construct model
    
outputs = mlp(X)

#define loss and optimizer
loss_out = tf.reduce_sum(tf.square(Y - outputs))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train_mlp = optimizer.minimize(loss_out)

# init variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    iterations = 3000
    epoch = 100
    for i in range(epoch):
        for n in range(iterations):
            rand_int = np.random.randint(len(processed_input)-1, size = batch_size)
            
            feed = {X:processed_input[rand_int, :], Y:processed_output[rand_int]}
            _, cost = sess.run([train_mlp, loss_out], feed_dict = feed)
            
            cost = cost/batch_size
            
            if n % 100 == 0:
                print("Epoch: {} \t Iteration: {} \t Loss: {}".format(i, n, cost))
            
            
            
            