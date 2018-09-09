import input_data
mnist = input_data.read_data_sets( "/tmp/data/", one_hot=true)

import tensorflow as tf

# set parameters
learning_rate = .01
training_iteration = 30
batch_size = 100
display_step = 2