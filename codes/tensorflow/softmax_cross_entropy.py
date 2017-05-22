import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir = 'datasets/mnist/'
input_layer_size = 784
output_layer_size = 10
learning_rate = 0.01
train_limit = 1000
batch_size = 100


def main():
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    W = tf.Variable(tf.zeros([input_layer_size, output_layer_size]), tf.float32)
    b = tf.Variable(tf.zeros([output_layer_size]))

    x = tf.placeholder(tf.float32, [None, input_layer_size])

    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, output_layer_size])

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # train the model
    for _ in range(train_limit):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, {x: batch_x, y_: batch_y})

    # test, accracy is about 0.9167
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1)), tf.float32))
    print('Test accuracy is %s.' % sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main()
