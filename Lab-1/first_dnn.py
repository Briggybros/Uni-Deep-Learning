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

np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
all_y = pd.get_dummies(data.iris_class)

# Dimensionality of input
n_x = len(all_x.columns)
# Dimensionality of output
n_y = len(all_y.columns)

# Training input
train_x = all_x[:100]
# Training output
train_y = all_y[:100]

# Testing input
test_x = all_x[100:150]
# Testing output
test_y = all_y[100:150]

# Placeholders (Just fill these from data at the time?)
x = tf.placeholder(tf.float32, shape=[None, n_x])
y = tf.placeholder(tf.float32, shape=[None, n_y])

# Number of nodes in each layer
layerSizes = [n_x, 10, 20, 10, n_y]

# Starting prediction
prediction = x

for idx, currentLayerSize in enumerate(layerSizes):
  # If we're at 0, we've already instantiated prediction
  if idx == 0: continue
  
  prevLayerSize = layerSizes[idx-1]
  W = tf.Variable(tf.truncated_normal([prevLayerSize, currentLayerSize], stddev=0.1))
  b = tf.Variable(tf.constant(0.1, shape=[currentLayerSize]))
  h = tf.nn.relu(tf.matmul(prediction, W) + b)

  # Update prediction
  prediction = h

# Make cost function
cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction, scope="Cost_Function")

# Optimise using Adagrad
optimiser = tf.train.AdagradOptimizer(0.01).minimize(cost)

sess.run(tf.global_variables_initializer())

groundTruth = np.argmax(test_y.values, axis=1)

# Do the training
for epoch in range(3000):
  sess.run([optimiser], feed_dict={x: train_x, y: train_y})
  if epoch % 100 == 0:
    out = sess.run(prediction, feed_dict={x: test_x, y: test_y}).tolist()
    predictions = np.argmax(out, axis=1)
    correct = np.sum(np.array(groundTruth == predictions, dtype=float))
    print("Accuracy of my first dnn at epoch %d is %d%%" % (epoch, correct * 100 / len(groundTruth)))
