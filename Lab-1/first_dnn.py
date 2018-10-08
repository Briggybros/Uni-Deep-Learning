############################################################
#                                                          #
#  Code for Lab 1: Your First Fully Connected Layer  #
#                                                          #
############################################################


import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd

sess = tf.Session()

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",",
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"])
#

np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
#
all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
#
all_y = pd.get_dummies(data.iris_class)
#

n_x = len(all_x.columns)
n_y = len(all_y.columns)

train_x = all_x[:100]
train_y = all_y[:100]

test_x = all_x[100:150]
test_y = all_y[100:150]

x = tf.placeholder(tf.float32, shape=[None, n_x])
y = tf.placeholder(tf.float32, shape=[None, n_y])

W = tf.Variable(tf.zeros((n_x, n_y)))
b = tf.Variable(tf.zeros(n_y))

prediction = tf.nn.softmax(tf.matmul(x, W) - b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis = 1))

optimiser = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess.run(tf.global_variables_initializer())

for epoch in range(10000):
  sess.run([optimiser], feed_dict={x: train_x, y: train_y}) 

out = sess.run(prediction, feed_dict={x: test_x, y: test_y}).tolist()
correct = 0

groundTruth = np.argmax(test_y.values, axis=1)

predictions = np.argmax(out, axis=1)

correct = np.sum(np.array(groundTruth == predictions, dtype=float))
print("%d%%" % (correct * 100 / len(groundTruth)))