#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/24 10:53
# @Author  : duocai

import numpy as np
import matplotlib.pyplot as plt


def plot_results(rmses, u, v):
    plt.figure()
    plt.plot(rmses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(u)
    plt.title("Users")
    plt.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(v)
    plt.title("Items")
    plt.axis("off")

    plt.show()

if __name__ == '__main__':
    results_dir = 'results/ml-100k/'
    file = '0.1_0.1_0.0004.npz'
    params = np.load(results_dir + file)

    plot_results(params['arr_0'], params['arr_1'], params['arr_2'])

