
import tensorflow as tf
from tensorflow.python.ops import rnn #, rnn_cell
from tensorflow.contrib.rnn import RNNCell
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import re
import random
import PrepareData
# from PrepareData import *
from collections import deque
# import tensorflow.python.debug as tf_debug

tf.reset_default_graph()
#  matplotlib inline

# Import MINST data
# work_train = "D:\\PycharmProjects\\ResearchWork\\Data\\Train"
work_train = "C:\\Users\\darid\\PycharmProjects\\Fall\\Data\\Train"
# work_test ="D:\\PycharmProjects\\ResearchWork\\Data\\Test"
work_test = "C:\\Users\\darid\\PycharmProjects\\Fall\\Data\\Test"
num = 0
# circle queue
ls_dir = os.listdir(work_train)
next_fn = deque(ls_dir, len(ls_dir))



# read all the data in a list : train_data
# train_data = []
# train_label = []
# for i in range(0, len(ls_dir)):
#      train_data.append(PrepareData.importData(work_train+'/'+ next_fn[i])[0])
#      train_label.append(PrepareData.importData(work_train+'/'+ next_fn[i])[1])
#
#
#
# next_train_data = deque(train_data, len(train_data))

def Next_data():
    # next_batch = next_train_data.rotate(-75*70)
    # return next_batch

    next_fn.rotate(-1)
    # print (next_fn[0])
    return PrepareData.importData(work_train+'\\'+ next_fn[0])


# Parameters
learning_rate = 0.000026 # 0.000026
training_iters = 200 #50000
batch_size = 1
display_step = 100

print (learning_rate)
# Network Parameters
n_input = 75 # data input  shape: 25(joints)*3(dimension)*80(frame)
n_steps = 75 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 8 # MNIST total classes (0-9 digits)

# tf Graph input

x = tf.placeholder("float", [None,n_steps, n_input])
y = tf.placeholder("float", [None,n_classes])

print (x,y)

# Define weights
with tf.name_scope('w'):
    weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    tf.summary.histogram(name='w', values=weights['out'])

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# tf.summary.histogram('biases', biases)

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    # print 'darid'

    x = tf.reshape(x, [-1, n_input])
    # print tf.shape(x)
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    # print tf.shape(x)
    # Define a lstm cell with tensorflow
    # lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0)
    # lstm_cell = RNNCell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2)
    lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell]*2)
    # Get lstm cell output

    outputs, states = tf.contrib.rnn.static_rnn(lstm_layers, x, dtype=tf.float32)
    # outputs, states = tf.nn.dynamic_rnn(lstm_layers, x, initial_state=init_state, time_major=False)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) #xutao 3/13/2017
    tf.summary.scalar('loss', cost) #xutao loss

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Evaluate model
results = tf.argmax(pred,1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
# init = tf.initialize_all_variables() # python 2.3
init = tf.global_variables_initializer()

# saver = tf.train.Saver()
#
# Launch the graph
with tf.Session() as sess:
    ##
    merged = tf.summary.merge_all() ##merge all the image
    writer = tf.summary.FileWriter('C:\\Users\\darid\\PycharmProjects\\Fall\\LSTM\\Graph', sess.graph)
    sess.run(init)
    step = 1
    # Keep training until reach max iterations


        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # # Darid Input Data
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # # Run optimization op (backprop)
    pn = 0
    ac_t = 0
    loss_t = 0
    i = 0
    while step * batch_size < training_iters:

        batch_x, batch_y = Next_data()

        # input data stream
        # if i >= len(train_data):
        #     i=0
        #
        # batch_x = train_data[i]
        #
        # batch_y = train_label[i]


        batch_x = np.asarray(batch_x)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        test = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        pr = sess.run(correct_pred, feed_dict={x: batch_x, y: batch_y})
        loss_c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        ac_c = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

        ac_t = ac_c + ac_t
        loss_t = loss_c + loss_t

        re = sess.run(results, feed_dict={x: batch_x})
        # print(re)
        # print(batch_y)
        # print(ac_c)


        ####original one
        # pr_s = str(pr[0])
        # # tf.print(test)
        # if pr_s=="True":
        #    pn = pn+1
        #####

        if step % display_step == 0:
        #     # Calculate batch accuracy

            print (ac_t)
            # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            # ac = float(pn) / display_step
            ac = ac_t/step
            loss = loss_t/step
            print( "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(ac))
            pn = 0
            ##########################
            rs = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(rs, step)
            #########################
        step += 1
        i += 1

    # print(sess.run(tf.global_variables()))# print all the variables
    # print(sess.run(weights))
    print ("Optimization Finished!")
# #
#     # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#     test_label = mnist.test.labels[:test_len]
#     print "Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: test_data, y: test_label})


    # Save the variables to disk by xutao 2017/3/19.
    # save_path = saver.save(sess, "C:\\Users\\darid\\PycharmProjects\\Fall\\LSTM\\Model7570")
    # print("Model saved in file: %s" % save_path)

    pn = 0
    tn = 0
    test_ac_t = 0

    for t_f in os.listdir(work_test):

        test_x, test_y = PrepareData.importData(work_test+'\\'+t_f)
        # print (test_x)
        test_x = np.asarray(test_x)

        test_x = test_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        # test = sess.run(optimizer, feed_dict={x: test_x, y: test_y})
        re = sess.run(results,feed_dict={x: test_x})
        print (re)
        # print(sess.run(weights))
        print (test_y)

        # print ("predictions", pr.eval(feed_dict={x: test_x}, session=sess))

        ################
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        print(test_acc)
        test_ac_t = test_acc + test_ac_t
        ##########################

        ################################
    print(test_ac_t/len(os.listdir(work_test)))

#######################################
#     pr = sess.run(correct_pred, feed_dict={x: test_x, y: test_y})
#     pr_s = str(pr[0])
#     print(pr_s)
#     if pr_s == "True":
#         pn = pn + 1
#
#     tn = tn+1
# print (float(pn)/tn)


