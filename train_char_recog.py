import tensorflow as tf
import numpy as np
import os
import pickle
import random

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


def pick_data(num,raw_data_list,raw_out_list):
    if num >= len(raw_data_list):
        return raw_data_list,raw_out_list

    data_res = []
    out_res = []
    # print("len(raw_data_list)",len(raw_data_list))
    # print("len(raw_out_list)", len(raw_out_list))
    for i in range(num):
        rid = random.randrange(0,len(raw_data_list)-1)
        # print(rid)
        data_res.append(raw_data_list[rid])
        out_res.append(raw_out_list[rid])

    return data_res, out_res


def compute_accuracy(in_test_data,out_test_data):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: in_test_data})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(out_test_data,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy, feed_dict={xs:in_test_data,ys:out_test_data})
    return result


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


# Load data
file_list = os.listdir(os.path.join(os.path.curdir,"data"))
train_data = []
train_out = []
test_data = []
test_out = []

for file in file_list:
    if '_train_' in file:
        with open(os.path.join(os.curdir,"data",file),'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                train_out.append(d["char"])
                train_data.append(d["data"])

    elif '_test_' in file:
        with open(os.path.join(os.curdir,"data",file), 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                test_out.append(d["char"])
                test_data.append(d["data"])


for i in range(2000):
    patch_data, patch_out = pick_data(100, train_data, train_out)
    sess.run(train_step,feed_dict={xs:patch_data,ys:patch_out})
    if i % 50 == 0:
        print(compute_accuracy(test_data,test_out))
save_path=saver.save(sess, os.path.join("network","save.ckpt"))
print("Save to path",save_path)
