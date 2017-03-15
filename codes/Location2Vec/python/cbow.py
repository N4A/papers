# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:07:40 2017

@author: duocai
"""

import numpy as np
import json
import math
import datetime
import random
import threading

#控制内积范围，超过范围的再加sigmoid很小，可以丢弃
MAX_EXP = 8

# 向量维度
layer_size = 100
# 单元location or word..个数
unit_num = 182968
# 单元向量表
vec_table = np.random.uniform(-1,1,layer_size*unit_num).reshape(unit_num,layer_size)
# 参数表, 参数个数即内部节点个数为叶节点个数（unit_num）-1, 初始化为正态分布取值
para_table =  np.random.normal(0.0,1,layer_size*(unit_num-1)).reshape(unit_num-1,layer_size)
# 所有用户，相当于所有文章
users = json.load(open('../user_new.json'))
# 所有地点，相当于所有单词
locats = json.load(open('../location_tree.v3.json'))
if len(locats.keys()) > unit_num:
    print('地点超出范围')
# window size
window = 5
# 学习率
start_alpha = 0.03
# 开始时间
start_time = datetime.datetime.now()

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def show_message():
    print('vector of location 0:',vec_table[0])

# 训练index指定的用户
def train(index):
    global layer_size
    global window
    global MAX_EXP
    global start_alpha
    global start_time
    sent = users[index]['locations']#文章，用户访问地点序列
    sent_len = len(sent)
    neul = np.zeros(layer_size)#隐藏层
    neule = np.zeros(layer_size)# 记录隐藏层累积变化量
    for pos in range(sent_len):
        loc_id = sent[0] # 地点id string
        if (loc_id not in locats):#不存在该地点则忽略
            continue
        # 计算context向量和
        num = random.randint(0,window)#随机起点，并不是严格从0开始
        start = num
        while start < 2 * window + 1 :
            cur_pos = pos - window + start
            #左右几个词不包含当前词
            # or 现在位置超过范围，则跳过
            if start == window or cur_pos < 0 or cur_pos >= sent_len:
                start+=1
                continue
            # 计算context向量和
            neul += vec_table[int(sent[cur_pos])]
            start+=1
        #利用霍夫曼树计算
        loc = locats[loc_id] # 当前地点
        points = loc['points']
        codes = loc['codes']
        codes_len = len(codes)
        for i in range(codes_len):
            # 计算内积
            dot = np.dot(neul,para_table[int(points[i])])
            ##内积不在范围内直接丢弃
            if dot > MAX_EXP or dot < -MAX_EXP:
                continue
            ##simoid
            dot = sigmoid(dot)
            # 偏导数的公用部分*学习率alpha
            g = (1 - int(codes[i]) - dot)*start_alpha
            
            # 反向更新参数
            # 先更新隐藏层
            neule += g*para_table[int(points[i])]
            # 后更新参数
            para_table[int(points[i])] += g*neul
                       
        ## 将更新传递到词向量
        start = num
        while start < 2 * window + 1 :
            cur_pos = pos - window + start
            #左右几个词不包含当前词
            # or 现在位置超过范围，则跳过
            if start == window or cur_pos < 0 or cur_pos >= sent_len:
                start+=1
                continue
            # 更新词向量
            vec_table[int(sent[cur_pos])] += neule
            start+=1
    # 输出现在的时间
    if index%200 == 0:
        print('user: ', index, ',time: ', (datetime.datetime.now() - start_time).seconds, 's')            
        show_message()
        
## 利用多线程训练
threads = []
for i in range(len(users)):
    threads.append(threading.Thread(target=train,args=(i,)))
## run
for t in threads:
    t.setDaemon(True)
    t.start()
t.join()#等待子线程结束