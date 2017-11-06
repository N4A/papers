#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/30 12:44
# @Author  : duocai

import numpy as np
import tensorflow as tf


def pmf(trains, tests, sigma=0.0001, lamb_u=0.01, lamb_v=0.001, latent_d=10, learning_r=0.0003, limit=1000000,
        verbose=True):
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

    # declare ratings's placeholder
    r = tf.placeholder(tf.int32, [3])

    # init latent factors using Gaussian distribution
    U = tf.Variable(tf.random_normal([user_num, latent_d], stddev=sigma / lamb_u), tf.float32)
    V = tf.Variable(tf.random_normal([item_num, latent_d], stddev=sigma / lamb_v), tf.float32)

    # loss = tf.reduce_sum(I * tf.square(R - tf.matmul(U, tf.matrix_transpose(V)))) \
    #        + lambU * tf.reduce_sum(tf.square(U)) \
    #        + lambV * tf.reduce_sum(tf.square(V))
    err = tf.square(tf.cast(r[2], tf.float32) - tf.reduce_sum(tf.multiply(U[r[0] - 1], V[r[1] - 1])))
    loss = err + lamb_u * tf.reduce_sum(tf.square(U[r[0] - 1])) + lamb_v * tf.reduce_sum(tf.square(V[r[1] - 1]))

    optimizer = tf.train.GradientDescentOptimizer(learning_r)
    train = optimizer.minimize(loss)

    # test accuracy
    TL = tf.constant(len(tests), tf.float32)
    # change to matrix
    test_ratings = np.zeros([user_num, item_num])
    test_identity = np.zeros([user_num, item_num])
    for rating in tests:
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
    Us = []
    Vs = []
    for i in range(limit):
        ratings_len = len(trains)
        rating_id = np.random.randint(0, ratings_len)
        # print(tf.constant(trains[rating_id]))
        sess.run(train, feed_dict={r: trains[rating_id]})
        if i % 10000 is 0:
            print("iteration %d, loss is %f" % (i, loss.eval(feed_dict={r: trains[rating_id]})))
            val = rmse.eval()
            print("rmse: %f" % val)
            rmses.append(val)
            Us.append(U.eval())
            Vs.append(V.eval())
    # accuracy
    print("rmse: %f" % rmse.eval())

    return rmses, Us, Vs


def output_result(path, rmses, U, V):
    np.savez(path, [rmses, U, V])


# test
if __name__ == '__main__':
    data_dir = 'datasets/ml-100k/'
    results_dir = 'results/ml-100k/'

    train = np.loadtxt(data_dir + 'u1.base', dtype=np.int32)[:, :3]  # cut timestamp
    test = np.loadtxt(data_dir + 'u1.test', dtype=np.int32)[:, :3]  # cut timestamp

    # best:
    # u1 rmse: 0.933554,
    # u2 rmse: 0.925295
    # u3 rmse: 0.920491
    sigma = 1e-4
    latent_ds = [20]
    lamb_us = [0.003]
    lamb_vs = [0.005]
    learn_rs = [3e-5]
    limit = 10000000

    for lamb_u in lamb_us:
        for lamb_v in lamb_vs:
            for learn_r in learn_rs:
                for latent_d in latent_ds:
                    path = results_dir + "pmf" + "_" + str(lamb_u) + "_" + str(lamb_v) + "_" + str(learn_r) + ".npz"
                    rmses, Us, Vs = pmf(train, test, sigma=sigma, latent_d=latent_d, lamb_u=lamb_u, lamb_v=lamb_v,
                                        learning_r=learn_r, limit=limit)
                    output_result(path, rmses, Us, Vs)
