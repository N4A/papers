#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/14 19:12
# @Author  : duocai

import numpy as np


class PMF(object):
    def __init__(self, trains, batch_size=3000, latent_d=20, lamb_u=0.01, lamb_v=0.09, learning_r=.02):
        self.latent_d = latent_d
        self.learning_r = learning_r
        self.lamb_u = lamb_u
        self.lamb_v = lamb_v
        self.batch_size = batch_size

        self.num_users = int(np.max(trains[:, 0]) + 1)
        self.num_items = int(np.max(trains[:, 1]) + 1)
        print('user numbers:', self.num_users - 1)
        print('item numbers:', self.num_items - 1)

        self.trains = trains

        self.U = np.random.random((self.num_users, self.latent_d))
        self.V = np.random.random((self.num_items, self.latent_d))

    def update(self):
        # SGD batch
        update_u = np.zeros([self.num_users, self.latent_d])
        update_v = np.zeros([self.num_items, self.latent_d])
        for k in range(self.batch_size):
            ratings_len = len(self.trains)
            rating_id = np.random.randint(0, ratings_len)
            (i, j, rating) = self.trains[rating_id]
            r_err = rating - np.dot(self.U[i], self.V[j])
            update_u[i] += self.learning_r * (self.lamb_u * self.U[i] - self.V[j] * r_err)
            update_v[j] += self.learning_r * (self.lamb_v * self.V[j] - self.U[i] * r_err)

        # update
        self.U -= update_u
        self.V -= update_v


def rmse(test_data, U, V):
    """
        Calculate root mean squared error. Ignoring missing values in the test data.
    """
    predicted = np.dot(U, V.transpose())
    N = len(test_data)  # number of non-missing values
    sqerror = 0
    for test_tuple in test_data:
        sqerror += abs(test_tuple[2] - predicted[test_tuple[0]][test_tuple[1]]) ** 2
    mse = sqerror / N  # mean squared error
    return np.sqrt(mse)


if __name__ == "__main__":
    data_dir = 'datasets/ml-100k/'
    results_dir = 'results/ml-100k/'

    train = np.loadtxt(data_dir + 'u1.base', dtype=np.int32)[:, :3]  # cut timestamp
    test = np.loadtxt(data_dir + 'u1.test', dtype=np.int32)[:, :3]  # cut timestamp

    print('rating numbers: ', len(train))

    pmf = PMF(train)

    time = 0
    while time <= 1e7:
        pmf.update()
        print("Iteration: %d, rmse: " % time, rmse(test, pmf.U, pmf.V))
        time += 1
