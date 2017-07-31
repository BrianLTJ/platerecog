import tensorflow as tf
import numpy as np
import os
import random
import argparse
import cv2
char_list = ["3","4","5","7","a","d","k","l","q","v"]

# %matplotlib inline
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope("weight"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')

        with tf.name_scope("bias"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,
                                 name='b')  # Bias value is recommended to be larger than 0
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

    return outputs


xs = tf.placeholder(tf.float32,[None,400])
ys = tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs,400,10,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

saver.restore(sess, os.path.join("network","save.ckpt"))


parser = argparse.ArgumentParser()
parser.add_argument("--img",type=str,default="")
img_addr, unknownargs = parser.parse_known_args()

raw_img = cv2.imread(img_addr.img)
ret, thr = cv2.threshold(raw_img,127,255,cv2.THRESH_BINARY)

thr_list = thr.tolist()
final_mat = []
for line_item in thr_list:
    for item in line_item:
        if item[0]*0.299+item[1]*0.587+item[2]*0.114 >= 180:
            final_mat.append(1)
        else:
            final_mat.append(0)

pred = sess.run(prediction,feed_dict={xs:[final_mat]})
print("Prediction Results")
print("ID","Possibility")
for i in range(len(pred[0])):
    print(i,float(pred[0][i])*100,"%")

id = sess.run(tf.argmax(pred,1))
id = id[0]
print("The Most Possible Char:",char_list[id])



