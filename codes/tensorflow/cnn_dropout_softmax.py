# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

data_dir = 'datasets/mnist/'
input_img_width = 28
input_size = 784  # 28*28
# output layer size
output_size = 10


# super parameters
batch_size = 100
train_limit = 20000
# convolution and pooling 1
patch_size_conv1 = 5
feature_num_conv1 = 32
stride_conv = 1
padding_conv = 'VALID'
size_pool = 2
stride_pool = 2
padding_pool = 'VALID'
# convolution and pooling 2
feature_num_conv2 = 64
patch_size_conv2 = 5
# stride_conv2 = 1
# padding_conv2 = 'VALID'
# stride_pool2 = 2
# padding_pool2 = 'VALID'
# full connection 1
size_fc1 = 1024  # similar like hidden layer size
# dropout
dropout_p = 0.38
# dropout2_p = 0.4
offset = 0  # batch window offset


# convolutional nerual net
# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# init bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, stride_conv, stride_conv, 1], padding=padding_conv)


# max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, size_pool, size_pool, 1],
                          strides=[1, stride_pool, stride_pool, 1], padding=padding_pool)


# main function
def main():
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    # input data
    x = tf.placeholder(tf.float32, [None, input_size])

    # output labels
    y_ = tf.placeholder(tf.float32, [None, output_size])

    # First Convolutional Layer
    W_conv1 = weight_variable([patch_size_conv1, patch_size_conv1, 1, feature_num_conv1])
    b_conv1 = bias_variable([feature_num_conv1])
    x_image = tf.reshape(x, [-1, input_img_width, input_img_width, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([patch_size_conv2, patch_size_conv2, feature_num_conv1, feature_num_conv2])
    b_conv2 = bias_variable([feature_num_conv2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    # get image witdh now
    image_width = int(((input_img_width - patch_size_conv1 + 1) / 2 - patch_size_conv2 + 1) / 2)
    W_fc1 = weight_variable([image_width * image_width * feature_num_conv2, size_fc1])
    b_fc1 = bias_variable([size_fc1])

    # change to 1D
    h_pool2_flat = tf.reshape(h_pool2, [-1, image_width * image_width * feature_num_conv2])

    # dropout 1
    keep_prob = tf.placeholder(tf.float32)
    h_pool2_drop = tf.nn.dropout(h_pool2_flat, keep_prob)

    # get hidden layer output
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_drop, W_fc1) + b_fc1)

    # Dropout 2
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([size_fc1, output_size])
    b_fc2 = bias_variable([output_size])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train and evaluate model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # train
    for i in range(train_limit):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout_p})

    # test
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
