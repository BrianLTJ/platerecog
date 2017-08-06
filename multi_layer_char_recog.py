import tensorflow as tf
import numpy as np

# Parameters
learingRate = 0.5
batchSize = 300

# Network parameters
nn_input = 400
nn_hidden_1 = 30

nn_output = 10

# Network in/output
x = tf.placeholder(tf.float32,[None,nn_input])
y = tf.placeholder(tf.float32,[None,nn_output])

def initWeights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1))


def oneHiddlenLayer(x,h1,y)

