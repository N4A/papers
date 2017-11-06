#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/24 16:36
# @Author  : duocai


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def pmf(trains, tests, lamb_u=0.01, lamb_v=0.001, latent_d=10, learning_r=0.0003, limit=1000000, verbose=True):
    """
    give training sets to learn a pmf
    :param trains: training sets [[user_id, item_id, ratings],....]
    :param tests:  test set [[user_id, item_id, ratings],....]
    :param lamb_u: parameter of user prior
    :param lamb_v parameter of item prior
    :param latent_d the length of latent vectors
    :param learning_r learing rate
    :param limit max iteration times
    :param verbose decide whether to print some verbose message, default is Ture
    :return: 
    """

    # decide number of users and items
    user_num = int(np.max(trains[:, 0]))
    item_num = int(np.max(trains[:, 1]))
    if verbose:
        print("user numbers: %d" % user_num)
        print("item numbers: %d" % item_num)

    # init latent vectors using Gaussian distribution
    # latent
    sigma = 0.001
    U = tf.Variable(tf.random_normal([user_num, latent_d], stddev=sigma / lamb_u), tf.float32)
    V = tf.Variable(tf.random_normal([item_num, latent_d], stddev=sigma / lamb_v), tf.float32)

    loss = 0
    for rating in trains:
        loss += tf.square(tf.matmul(U[rating[0]], V[rating[1]], transpose_b=True) - rating[2])
    # add regulization term
    loss += lamb_u * tf.reduce_sum(tf.square(U)) + lamb_v * tf.reduce_sum(tf.square(V))

    optimizer = tf.train.GradientDescentOptimizer(learning_r)
    train = optimizer.minimize(loss)

    # test accuracy
    TL = tf.constant(len(tests), tf.float32)
    # change to matrix
    test_ratings = np.zeros([user_num, item_num])
    test_identity = np.zeros([user_num, item_num])
    for rating in trains:
        test_ratings[rating[0] - 1, rating[1] - 1] = rating[2]
        test_identity[rating[0] - 1, rating[1] - 1] = 1
    TR = tf.constant(test_ratings, tf.float32)
    TI = tf.constant(test_identity, tf.float32)
    rmse = tf.sqrt(tf.reduce_sum(TI * tf.square(TR - tf.matmul(U, tf.matrix_transpose(V)))) / TL)

    # get session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # start traing
    rmses = []
    for i in range(limit):
        sess.run(train)
        if i % 500 is 0:
            print("iteration %d, loss is %f" % (i, loss.eval()))
            val = rmse.eval()
            print("rmse: %f" % val)
            rmses.append(val)
    # accuracy
    print("rmse: %f" % rmse.eval())

    return rmses, U.eval(), V.eval()


def plot_results(rmses, U, V):
    plt.figure()
    plt.plot(rmses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(U)
    plt.title("Users")
    plt.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(V)
    plt.title("Items")
    plt.axis("off")

    plt.show()


def output_result(path, rmses, U, V):
    np.savez(path, [rmses, U, V])


# test
if __name__ == '__main__':
    data_dir = 'datasets/ml-100k/'
    results_dir = 'results/ml-100k/'

    train = np.loadtxt(data_dir + 'u1.base', dtype=np.int32)[:, :3]  # cut timestamp
    test = np.loadtxt(data_dir + 'u1.test', dtype=np.int32)[:, :3]  # cut timestamp

    lamb_us = [0.1, 0.03, 0.001, 0.0003, 0.0001]
    lamb_vs = [0.1, 0.03, 0.001, 0.0003, 0.0001]
    learn_rs = [0.0004, 0.0003, 0.0002, 0.0001]
    limit = 2000000

    for lamb_u in lamb_us:
        for lamb_v in lamb_vs:
            for learn_r in learn_rs:
                path = results_dir + "pmf" + "_" + str(lamb_u) + "_" + str(lamb_v) + "_" + str(learn_r) + ".npz"
                rmses, U, V = pmf(train, test, lamb_u=lamb_u, lamb_v=lamb_v, learning_r=learn_r, limit=limit)
                output_result(path, rmses, U, V)

                # plot result
                # plot_results(rmses, U, V)
