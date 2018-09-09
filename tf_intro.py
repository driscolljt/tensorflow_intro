# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( "/tmp/data/", one_hot=true)

import tensorflow as tf

# set parameters
learning_rate = .01
training_iteration = 30
batch_size = 100
display_step = 2

# TF graph input
x = tf.placeholder( "float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder( "float", [None, 10]) # 0-9 digits recognition => 10 classes

# create model

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    # construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) #softmax

# add summary ops to collect data
w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

# more name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    # minimize error using cross entropy
    # cross entropy
    cost_function = tf.reduce_sum(y*tf.log(model))
    # create a summary to monitor the cost function
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# init variables
init = tf.initialize_all_variables()

# merge all summaries into a single operator
merged_summary_op = tf.merge_all_summaries()

# launch the graph
# todo: continue from siraj tutorial tensorflow in 5 minutes