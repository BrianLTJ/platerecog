import numpy as np
import tensorflow as tf
import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def model(X, w_layer_1, w_layer_2, w_layer_3, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
    hidden_1 = tf.nn.relu(tf.matmul(X, w_layer_1))

    hidden_1 = tf.nn.dropout(hidden_1, p_keep_hidden)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w_layer_2))

    hidden_2 = tf.nn.dropout(hidden_2, p_keep_hidden)

    return tf.matmul(hidden_2, w_layer_3)

# 导入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

# 在该模型中我们一共有4层，一个输入层，两个隐藏层，一个输出层
# 定义输入层到第一个隐藏层之间的连接矩阵
w_layer_1 = init_weights([784, 625])

# 定义第一个隐藏层到第二个隐藏层之间的连接矩阵
w_layer_2 = init_weights([625, 625])

# 定义第二个隐藏层到输出层之间的连接矩阵
w_layer_3 = init_weights([625, 10])

# dropout 系数
# 定义有多少有效的神经元将作为输入神经元，比如 p_keep_intput = 0.8，那么只有80%的神经元将作为输入
p_keep_input = tf.placeholder("float")

# 定义有多少的有效神经元将在隐藏层被激活
p_keep_hidden = tf.placeholder("float")

# 构建模型
py_x = model(X, w_layer_1, w_layer_2, w_layer_3, p_keep_input, p_keep_hidden)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:

    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict = {X: trX[start:end], Y: trY[start:end],
                                            p_keep_input: 0.8, p_keep_hidden: 0.5})
        print i, np.mean(np.argmax(teY, axis = 1) == sess.run(predict_op,
                        feed_dict = {X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0}))