#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/24 18:03
# @Author  : duocai
from adam.PMF import PMF
import numpy as np

if __name__ == "__main__":
    data_dir = '../datasets/ml-100k/'
    results_dir = '../results/ml-100k/'

    train = np.loadtxt(data_dir + 'u1.base', dtype=np.int32)[:, :3]  # cut timestamp
    test = np.loadtxt(data_dir + 'u1.test', dtype=np.int32)[:, :3]  # cut timestamp

    pmf = PMF()
    pmf.fit(train, test)
