import tensorflow as tf


def linear_model(train_x, train_y):
    # Model parameters
    W = tf.Variable(.3, tf.float32)
    b = tf.Variable(-.3, tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    y_ = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(y_ - y)) # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)  # reset values to wrong
    # training loop
    for i in range(1000):
      sess.run(train, {x: train_x, y: train_y})

    # evaluate training accuracy
    return sess.run([W, b, loss], {x: train_x, y: train_y})


# test
if __name__ == '__main__':
    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    W, b, loss = linear_model(train_x=x_train, train_y=y_train)
    print("y = %s * x + %s, and loss is %s" % (W, b, loss))
