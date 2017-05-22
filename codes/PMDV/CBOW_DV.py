# -*- coding: utf-8 -*-
"""
Created on Tue 3/17 17:07:40 2017

@author: duocai
"""

import numpy as np
import json
import math
import datetime
import random
import doc_manager as dm

# super parameter
# window size
window = 5
# 学习率
start_alpha = 0.03
# 控制内积范围，超过范围的再加sigmoid很小，可以丢弃
MAX_EXP = 10
# 向量维度
layer_size = 20

#
temp_update = np.zeros(layer_size)

# internal parameter
doc_num = 0
unit_num = 0
# 单元向量表
word_vec_table = {}
# 文档向量表
doc_vec_table = {}
# 参数表, 参数个数即内部节点个数为叶节点个数（unit_num）-1, 初始化为正态分布取值
para_table = {}
# 所有用户，相当于所有文章
docs = {}
# 所有地点，相当于所有单词
words_tree = {}
# the index of the doc in the doc_vec_table
doc_index = {}
# the index of the word in the word_vec_table
word_index = {}
# 开始时间
start_time = datetime.datetime.now()
# iteration time
train_num = 0


def init_data(doc_path, words_path):
    if not(doc_path or words_path):
        print('please fill doc_path and words_path')
        exit()
    global docs
    global words_tree
    # default 'data/movielen/handled/doc.txt'
    docs = dm.read_handled_data(doc_path)
    # default 'data/movielen/handled/tree.txt'
    words_tree = dm.read_handled_data(words_path)


def init_basic_parameter(p_layer_size=layer_size, p_window_size=window,
                         p_learn_alpha=start_alpha, p_MAX_EXP=MAX_EXP):
    global docs
    global words_tree
    if docs == {} or words_tree == {}:
        print('Err: please init data with doc_path and words_path first')
        exit()

    global layer_size
    global window
    global MAX_EXP
    global start_alpha
    layer_size = p_layer_size
    window = p_window_size
    MAX_EXP = p_MAX_EXP
    start_alpha = p_learn_alpha

    global unit_num
    global doc_num
    unit_num = len(words_tree)
    doc_num = len(docs)
    global para_table
    global doc_vec_table
    global word_vec_table
    # 单元向量表
    word_vec_table = np.random.uniform(0, 1, layer_size * unit_num).reshape(unit_num, layer_size)
    # 参数表, 参数个数即内部节点个数为叶节点个数（unit_num）-1, 初始化为正态分布取值
    para_table = np.random.normal(0.0, 1, layer_size * (unit_num - 1)).reshape(unit_num - 1, layer_size)
    # 文档向量表
    doc_vec_table = np.random.uniform(0, 1, layer_size * doc_num).reshape(doc_num, layer_size)

    # map from the key of words, docs to index of vector table
    index = 0
    for word in words_tree.keys():
        word_index[word] = index
        index += 1
    index = 0
    for doc in docs.keys():
        doc_index[doc] = index
        index += 1


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def normalize(vec):
    average, std = np.average(vec), np.std(vec)
    return (vec - average) / std


# show message of the training process
def show_message():
    def by_similar(word_s):
        return word_s[1]
    # check for similar
    num = 0
    words = ['will', 'to', 'like', 'good', 'eight', 'or', 'dog', 'movie']
    for word in words:
        if word not in words_tree:
            continue
        vec = word_vec_table[word_index[word]]
        similar = []
        for other in words_tree.keys():
            other_vec = word_vec_table[word_index[other]]
            similar.append((other, np.dot(vec, other_vec)))
        similar_sorted = sorted(similar, key=by_similar, reverse=True)
        print(word, ':', list(map(lambda x: x[0], similar_sorted[0:10])))
        num += 1
        if num > 10:
            break


# 训练doc_key指定的文章
def DV_Enhanced_CBOW(doc_key):

    if doc_key not in docs.keys():
        return

    sent = docs[doc_key][0].split('\t')  # 文章(多篇文章只取第一篇)，用户访问地点序列
    sent_len = len(sent)

    # start train
    for pos in range(sent_len):
        neul = np.zeros(layer_size)  # 隐藏层
        neule = np.zeros(layer_size)  # 记录隐藏层累积变化量

        word = sent[pos]  # 地点id string
        if word not in words_tree:  # 不存在该词则忽略
            continue

        # 计算context向量和
        # 先加上本段文章向量，与pos无关
        neul += doc_vec_table[doc_index[doc_key]]
        num = random.randint(0, window)  # 随机起点，并不是严格从0开始
        start = num
        while start < 2 * window + 1:
            cur_pos = pos - window + start
            # 左右几个词不包含当前词
            # or 现在位置超过范围，则跳过  , and ignore ''
            if start == window or cur_pos < 0 or cur_pos >= sent_len or sent[cur_pos] == '':
                start += 1
                continue
            # 计算context向量和
            neul += word_vec_table[word_index[sent[cur_pos]]]
            start += 1

        # 利用霍夫曼树计算
        word_in_tree = words_tree[word]  # 当前地点
        points = word_in_tree['points']
        codes = word_in_tree['codes']
        codes_len = len(codes)
        for layer_index in range(codes_len):
            # 计算内积
            dot = np.dot(neul, para_table[points[layer_index]])
            # 内积不在范围内直接丢弃
            if dot > MAX_EXP or dot < -MAX_EXP:
                continue
            # simoid
            dot = sigmoid(dot)
            # 偏导数的公用部分*学习率alpha
            g = (1 - codes[layer_index] - dot) * start_alpha

            # 反向更新参数
            # 先更新隐藏层
            neule += g * para_table[points[layer_index]]
            # 后更新参数
            para_table[points[layer_index]] += g * neul
            # normalize
            para_table[points[layer_index]] = normalize(para_table[points[layer_index]])

        # 更新doc vector
        doc_vec_index = doc_index[doc_key]
        doc_vec_table[doc_vec_index] += neule
        # normalize
        doc_vec_table[doc_vec_index] = normalize(doc_vec_table[doc_vec_index])
        # 将更新传递到词向量
        start = num
        while start < 2 * window + 1:
            cur_pos = pos - window + start
            # 左右几个词不包含当前词
            # or 现在位置超过范围，则跳过 and ignore ''
            if start == window or cur_pos < 0 or cur_pos >= sent_len or sent[cur_pos] == '':
                start += 1
                continue
            # 更新词向量
            vec_index = word_index[sent[cur_pos]]
            word_vec_table[vec_index] += neule
            # 修正词向量,normalize
            word_vec_table[vec_index] = normalize(word_vec_table[vec_index])
            start += 1

    # 输出现在的时间
    global train_num
    train_num += 1
    if train_num % 500 == 0:
        print('Iteration: ', train_num, ',time: ', (datetime.datetime.now() - start_time).seconds, 's')
        show_message()


def ndarray_to_json(ndarr):
    ret = []
    for vec in ndarr:
        ret.append(list(map(lambda x: float(x), list(vec))))
    return json.dumps(ret)


def init():
    init_data('data/movielen/handled/doc.txt', 'data/movielen/handled/tree.txt')
    init_basic_parameter(layer_size, window, start_alpha, MAX_EXP)

def output():
    # output vec
    word = open('data/movielen/vector/word_vec.txt', 'w')
    doc = open('data/movielen/vector/doc_vec.txt', 'w')
    doc_index = open('data/movielen/vector/doc_index.txt', 'w')
    word.write(ndarray_to_json(word_vec_table))
    doc.write(ndarray_to_json(doc_vec_table))
    doc_index.write(doc_index)
    word.close()
    doc.close()
# test
if __name__ == '__main__':
    init()
    for iter_num in range(40):
        for doc_key in docs.keys():
            DV_Enhanced_CBOW(doc_key)
    output()
