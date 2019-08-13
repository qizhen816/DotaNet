from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import gzip
import time
import pickle
import tensorflow as tf
import os
import numpy as np
import random
def testtf():
    f = gzip.open('DotaTM0724raw.set', 'rb')
    data_set, label_set = pickle.load(f, encoding='latin1')
    f.close()
    XX = data_set[0:5]

    X = tf.placeholder("float", [5, 2046],name= 'X')

    X1 = tf.reshape(X[:, 0:1020], [-1, 20, 51])
    X2 = tf.reshape(X[:, 1020:2040], [-1, 20, 51])
    x = tf.concat([X1, X2], 1) #2040
    #X3 = X[:, 0:3]
    #X4 = X[:, 1023:1026]
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x_reshape = tf.reshape(X1, [-1, 20, 51]) #20场比赛 包含两个队的胜负 102
    #X1 = x_reshape[:, :, 0:50] #一个队
    Xyes1 = x_reshape[:,:,50]
    x_cnn = tf.reshape(x_reshape, shape=[-1, 20,51,1])  # 输入尺寸

    #X2 = x_reshape[:, :, 51:101] #另一个队
    #Xyes2 = x_reshape[:, :, 101] #第二个队的胜负
    #x0 = tf.concat([X1, X2], 2)
    #x_rnn = tf.reshape(x0,[-1,100])[:,0]
    #X_results = tf.concat([Xyes1, Xyes2], 1)
    #fc1 = tf.reshape(X_results, [-1, tf.Variable(tf.random_normal([40, 256])).get_shape().as_list()[0]])

    # 初始化变量
    init = tf.initialize_all_variables()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    print('20场比赛胜负1：',list(sess.run(Xyes1,feed_dict={X:XX})[0]))
    #print('20场比赛胜负2：',list(sess.run(Xyes2,feed_dict={X:XX})[0]))
    #print('加起来：',list(sess.run(fc1,feed_dict={X:XX})[0][0:20]),list(sess.run(fc1,feed_dict={X:XX})[0][20:40]))

    #print('队伍数据：',len(list(sess.run(x,feed_dict={X:XX})[0])),list(sess.run(x,feed_dict={X:XX})[0]))
    print('20x51（1）:',list(sess.run(x_reshape,feed_dict={X:XX})[0][0][0:51]))
    #print('20x102（2）:',list(sess.run(x_reshape,feed_dict={X:XX})[0][0][51:102]))

    #print('改变为时序，两个20*50拼接（1）：',list(sess.run(x0,feed_dict={X:XX})[0][0][0:50]))
    #print('改变为时序，两个20*50拼接（2）：',list(sess.run(x0,feed_dict={X:XX})[0][0][50:100]))

    print('CNN reshape:：',list(sess.run(x_cnn,feed_dict={X:XX})))


#testtf()
visit0828 = []

with open('visit0828.txt') as f:
    for line in f:
        if len(line.split())>1 and line.split()[1] == 'mark':
            visit0828.append(line.split()[0])
count = 0
with open('visit0724.txt') as f:
    for line in f:
        if len(line.split())>1 and line.split()[1] == 'mark':
            if line.split()[0] in visit0828:
                count += 1
print(count)