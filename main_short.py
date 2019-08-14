from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from data_set import Reader
from data_set import PreProcess
import tensorflow as tf
import numpy as np
from config import *
from data_set.getDotaData import *

FLAGS = tf.app.flags.FLAGS



def weight_variable(shape,lamda= 0.001):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lamda)(var))
    # 把正则化加入集合losses里面
    return var

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x,w = 4):
    return tf.nn.max_pool(x, ksize=[1,w,w,1], strides=[1,w,w,1], padding='SAME')

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True): #test????????
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name,
                        )

def resize(X,num):
    X_ = tf.reshape(X[:], [-1, FLAGS.estep, FLAGS.esize])
    if num ==30:
        X_data = tf.reshape(X_[:, :, 0:-1], [-1, FLAGS.estep, 5, 10]) #batch 时序 选手 数据
        X_tmp = tf.reshape(X_data[:, :, :, 0], [-1, FLAGS.estep, 5, 1])
        X_in = tf.concat([X_tmp, X_data[:, :, :, 2:5]], 3)
        X_in = tf.concat([X_in, X_data[:, :, :, 7:8]], 3)
        #X_in = tf.concat([X_in, X_data[:, :, :, 8:9]], 3)
        X_in = tf.reshape(X_in, [-1, FLAGS.estep, 25])
    else:
        X_in = tf.reshape(X_[:, :, 0:-1], [-1, FLAGS.estep, 50])  # batch 时序 选手x数据
    return X_in,tf.reshape(X_[:, :, -1], shape=[-1, FLAGS.estep, 1])

def train():
    print(FLAGS.estep)

    mode = 'train'
    model_dir = './checkpoint/model.ckpt-2000'
    restore = False

    dota = Reader.read_all \
        (FLAGS.datadir, train_num=FLAGS.train_length, test_num=FLAGS.test_length, validation_num=FLAGS.Vali_length)

    with tf.name_scope('data'):
        X = tf.placeholder(dtype=tf.float32, shape=(FLAGS.ebatch, 2046), name="inputs")
        Y = tf.placeholder(dtype=tf.float32, shape=(FLAGS.ebatch, FLAGS.eout), name='label')

    keep_prob = tf.placeholder(tf.float32, name='dropout')
    bn_1 = batch_norm(name='bn1')
    bn_2 = batch_norm(name='bn2')

    X_1_,input_1_ext = resize(X[:, 0:FLAGS.estep * FLAGS.esize],30)
    X_2_,input_2_ext = resize(X[:, 1020:1020+FLAGS.estep * FLAGS.esize],30)


    FLAGS.esize = 25

    input_1 = tf.reshape(X_1_, shape=[-1, FLAGS.esize])
    input_2 = tf.reshape(X_2_, shape=[-1, FLAGS.esize])

    fc_1_w = weight_variable([FLAGS.esize,FLAGS.ecell-1])
    fc_1_b = bias_variable([FLAGS.ecell-1])
    rnn_1 = tf.matmul(input_1,fc_1_w)+fc_1_b
    # rnn_1 = tf.contrib.layers.batch_norm(rnn_1, scale=True, is_training=True, updates_collections=None)
    rnn_1 = tf.nn.relu(rnn_1)
    #rnn_1 = tf.nn.dropout(rnn_1, keep_prob)
    rnn_1 = tf.reshape(rnn_1, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
    rnn_1 = tf.concat([rnn_1, input_1_ext], 2)
    rnn_1 = tf.reshape(rnn_1, (-1, FLAGS.estep, FLAGS.ecell))

    if FLAGS.rnn == 'LSTM':
        cell_1 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_1 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_1 = tf.nn.rnn_cell.DropoutWrapper \
        (cell_1, output_keep_prob=keep_prob)#, input_keep_prob=keep_prob)  # drop_out
    init_state_1 = cell_1.zero_state(FLAGS.ebatch, dtype=tf.float32)
    X_1, _ = tf.nn.dynamic_rnn(cell_1, rnn_1, initial_state=init_state_1, scope='1')

    fc_2_w = weight_variable([FLAGS.esize,FLAGS.ecell-1])
    fc_2_b = bias_variable([FLAGS.ecell-1])
    rnn_2 =tf.matmul(input_2,fc_2_w)+fc_2_b
    # rnn_2 = tf.contrib.layers.batch_norm(rnn_2, scale=True, is_training=True, updates_collections=None)
    rnn_2 = tf.nn.relu(rnn_2)
    #rnn_2 = tf.nn.dropout(rnn_2, keep_prob)
    rnn_2 = tf.reshape(rnn_2, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
    rnn_2 = tf.concat([rnn_2, input_2_ext], 2)
    rnn_2 = tf.reshape(rnn_2, (-1, FLAGS.estep, FLAGS.ecell))
    if FLAGS.rnn == 'LSTM':
        cell_2 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_2 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_2 = tf.nn.rnn_cell.DropoutWrapper \
        (cell_2, output_keep_prob=keep_prob)#, input_keep_prob=keep_prob)  # drop_out
    init_state_2 = cell_2.zero_state(FLAGS.ebatch, dtype=tf.float32)
    X_2, _ = tf.nn.dynamic_rnn(cell_2, rnn_2, initial_state=init_state_2, scope='2')

    # X_1 = bn_1(X_1)
    # X_2 = bn_2(X_2)
    X_1 = tf.transpose(X_1, [1, 0, 2])[-1]
    X_2 = tf.transpose(X_2, [1, 0, 2])[-1]
    # X_12 = tf.concat([X_1,X_2],1)


    w_1 = weight_variable([FLAGS.ecell,FLAGS.eout])
    b_1 = bias_variable([FLAGS.eout])
    X_1_fc = tf.matmul(X_1, w_1) + b_1

    w_2 = weight_variable([FLAGS.ecell,FLAGS.eout])
    b_2 = bias_variable([FLAGS.eout])
    X_2_fc = tf.matmul(X_2, w_2) + b_2


    R_A = tf.reshape(X[:, -6], [FLAGS.ebatch, 1])/23
    R_B = tf.reshape(X[:, -3], [FLAGS.ebatch, 1])/23
    logits = R_A - R_B - (X_1_fc - X_2_fc)
    entropy = tf.square(tf.ones([FLAGS.ebatch, 1]) - Y - logits)

    # logits = X_1_fc - X_2_fc
    # entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)

    loss = tf.reduce_mean(entropy)
    # tf.add_to_collection("losses",loss)
    # loss_batches = tf.reduce_mean(entropy,1)
    # loss = tf.add_n(tf.get_collection('losses'))
    optimizer = tf.train.AdamOptimizer(FLAGS.lrate).minimize(loss)


    length_0 = tf.abs(tf.zeros([FLAGS.ebatch, 1])-logits)
    length_1 = tf.abs(tf.ones([FLAGS.ebatch, 1])-logits)
    logits = tf.concat([length_1,length_0], 1)
    logits_ = tf.reshape(tf.cast(tf.argmax(logits, 1), tf.float32), [FLAGS.ebatch, 1])
    acc = tf.equal(tf.ones([FLAGS.ebatch, 1]) -Y, logits_)
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))*100 # cast：转化格式

    # y_pred = tf.arg_max(logits, 1)
    # bool_pred = tf.equal(tf.arg_max(Y, 1), y_pred)
    # accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32)) * 100  # cast：转化格式

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
            if mode == 'train':
                max_epoch = FLAGS.epoch + 1
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                if restore == True:
                    saver.restore(sess, model_dir)
                vs = tf.trainable_variables()
                print('There are %d train_able_variables in the Graph: ' % len(vs))
                for v in vs:
                    print(v)
            else:
                max_epoch = 1
                FLAGS.dropout_rate = 1.0
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, model_dir)
            start_time = time.time()
            batch_account = 0
            batch_idxs_train = FLAGS.train_length // FLAGS.ebatch
            batch_idxs_vali = FLAGS.Vali_length // FLAGS.ebatch
            batch_idxs_test = FLAGS.test_length // FLAGS.ebatch

            for epoch in range(1,max_epoch+1):
                total_loss = 0.0
                total_acc = 0.0
                for index in range(1,batch_idxs_train+1):
                    X_batch, Y_batch = dota.train.next_batch(FLAGS.ebatch)
                    accbatch, _, loss_batch = sess.run([accuracy,optimizer,loss],
                                                      feed_dict={X: X_batch, Y: Y_batch, keep_prob: FLAGS.dropout_rate})
                    total_loss += loss_batch
                    total_acc += accbatch
                    batch_account += 1
                print('Batch loss at epoch {}: {:5.4f}, accuracy:{:5.4f}'
                      .format(epoch,total_loss/batch_idxs_train,total_acc/batch_idxs_train))
                if epoch == 1 or epoch %10 == 0:
                    with open('loss'+str(FLAGS.rnn)+str(FLAGS.estep)+'.txt',"a") as f:
                        f.write(str(epoch)+' '+str(total_loss/batch_idxs_train))
                        f.write('\n')
                    f.close()
                #validation
                if (epoch) % (10) == 0 or mode != 'train':
                    total_loss = 0.0
                    total_acc = 0.0
                    for index in range(1, batch_idxs_vali + 1):
                        X_batch, Y_batch = dota.validation.next_batch(FLAGS.ebatch)
                        accbatch, _, loss_batch, tst = sess.run([accuracy,optimizer, loss, logits],
                                                       feed_dict={X: X_batch, Y: Y_batch, keep_prob: FLAGS.dropout_rate})
                        total_loss += loss_batch
                        total_acc += accbatch
                        batch_account += 1
                    print('Validation loss at step {}:{:5.4f}, accuracy:{:5.4f}'
                          .format(epoch,total_loss/batch_idxs_vali,total_acc/batch_idxs_vali))
                    if mode == 'train':
                        saver.save(sess, './checkpoint/model.ckpt', epoch)
            print("Optimization Finished!")  # should be around 0.35 after 25 epochs
            print("Total time: {0} seconds".format(time.time() - start_time))
            duration = time.time() - start_time
            print(duration,duration/batch_account)
            total_loss = 0.0
            total_acc = 0.0
            for i in range(batch_idxs_test):
                X_batch, Y_batch = dota.test.next_batch(FLAGS.ebatch)
                accbatch, loss_batch = sess.run([accuracy, loss],
                                               feed_dict={X: X_batch, Y: Y_batch, keep_prob: 1.0})
                total_loss += loss_batch
                total_acc += accbatch
            print("TEST Loss{:5.4f}, accuracy:{:5.4f}"
                  .format(total_loss/batch_idxs_test,total_acc/batch_idxs_test))
    return  total_loss/batch_idxs_test,total_acc/batch_idxs_test

def get1data(ids = [-1],ass = [-1],bss = [-1]):
    tl = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data_set/TeamTop190813.txt'

    opener = newop()
    opener, payload, headres1 = lgin(opener)
    Teams, Visit = readFirstList(opener,TL = tl)
    dataname = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data0.txt'
    labelname = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/label0.txt'

    if ids[0] != -1:
        for id in ids:
            data, winlose, twoTeam = getRecentGame(opener, Teams, id,defaultF = 1)
            data = data + twoTeam[0][0:3] + twoTeam[1][0:3]  # 应该是[1:4]没有改 后面预处理再改
            writeData(dataname, data)
            writeData(labelname, winlose)
    elif ass[0]!= -1:
        for i in range(len(ass)):
            data,twoTeam = getRecentGameNoID(opener,[Teams[ass[i]-1][0],Teams[bss[i]-1][0]],Teams)
            data = data + twoTeam[0][0:3] + twoTeam[1][0:3]  # 应该是[1:4]没有改 后面预处理再改
            winlose = 999
            writeData(dataname, data)
            writeData(labelname, winlose)

    Obj = PreProcess.RawData([dataname],[labelname],tl)
    All = Obj.pre_RNN(old=True)
    All = Obj.change_length(All, changeorder=True, length=2046)
    All = Obj.reverse(All)

    print(All[0])
    return All

def predict():
    print(FLAGS.estep)

    model_dir = './checkpoint/model.ckpt-2000'

    a = [3,3,2]
    b = [2,6,6]
    all = get1data(ids = [-1],ass=a,bss=b) #4063572817,4065268827,4065211218
    FLAGS.ebatch = len(all[0])

    dota = Reader.read_all \
    (FLAGS.datadir, train_num=FLAGS.train_length, test_num=FLAGS.test_length, validation_num=FLAGS.Vali_length, All=all)

    with tf.name_scope('data'):
        X = tf.placeholder(dtype=tf.float32, shape=(FLAGS.ebatch, 2046), name="inputs")
        Y = tf.placeholder(dtype=tf.float32, shape=(FLAGS.ebatch, FLAGS.eout), name='label')

    keep_prob = tf.placeholder(tf.float32, name='dropout')
    bn_1 = batch_norm(name='bn1')
    bn_2 = batch_norm(name='bn2')

    X_1_,input_1_ext = resize(X[:, 0:FLAGS.estep * FLAGS.esize],30)
    X_2_,input_2_ext = resize(X[:, 1020:1020+FLAGS.estep * FLAGS.esize],30)


    FLAGS.esize = 25

    input_1 = tf.reshape(X_1_, shape=[-1, FLAGS.esize])
    input_2 = tf.reshape(X_2_, shape=[-1, FLAGS.esize])

    fc_1_w = weight_variable([FLAGS.esize,FLAGS.ecell-1])
    fc_1_b = bias_variable([FLAGS.ecell-1])
    rnn_1 = tf.matmul(input_1,fc_1_w)+fc_1_b
    # rnn_1 = tf.contrib.layers.batch_norm(rnn_1, scale=True, is_training=True, updates_collections=None)
    rnn_1 = tf.nn.relu(rnn_1)
    #rnn_1 = tf.nn.dropout(rnn_1, keep_prob)
    rnn_1 = tf.reshape(rnn_1, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
    rnn_1 = tf.concat([rnn_1, input_1_ext], 2)
    rnn_1 = tf.reshape(rnn_1, (-1, FLAGS.estep, FLAGS.ecell))

    if FLAGS.rnn == 'LSTM':
        cell_1 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_1 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_1 = tf.nn.rnn_cell.DropoutWrapper \
        (cell_1, output_keep_prob=keep_prob)#, input_keep_prob=keep_prob)  # drop_out
    init_state_1 = cell_1.zero_state(FLAGS.ebatch, dtype=tf.float32)
    X_1, _ = tf.nn.dynamic_rnn(cell_1, rnn_1, initial_state=init_state_1, scope='1')

    fc_2_w = weight_variable([FLAGS.esize,FLAGS.ecell-1])
    fc_2_b = bias_variable([FLAGS.ecell-1])
    rnn_2 =tf.matmul(input_2,fc_2_w)+fc_2_b
    # rnn_2 = tf.contrib.layers.batch_norm(rnn_2, scale=True, is_training=True, updates_collections=None)
    rnn_2 = tf.nn.relu(rnn_2)
    #rnn_2 = tf.nn.dropout(rnn_2, keep_prob)
    rnn_2 = tf.reshape(rnn_2, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
    rnn_2 = tf.concat([rnn_2, input_2_ext], 2)
    rnn_2 = tf.reshape(rnn_2, (-1, FLAGS.estep, FLAGS.ecell))
    if FLAGS.rnn == 'LSTM':
        cell_2 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_2 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_2 = tf.nn.rnn_cell.DropoutWrapper \
        (cell_2, output_keep_prob=keep_prob)#, input_keep_prob=keep_prob)  # drop_out
    init_state_2 = cell_2.zero_state(FLAGS.ebatch, dtype=tf.float32)
    X_2, _ = tf.nn.dynamic_rnn(cell_2, rnn_2, initial_state=init_state_2, scope='2')

    # X_1 = bn_1(X_1)
    # X _2 = bn_2(X_2)
    X_1 = tf.transpose(X_1, [1, 0, 2])[-1]
    X_2 = tf.transpose(X_2, [1, 0, 2])[-1]
    # X_12 = tf.concat([X_1,X_2],1)


    w_1 = weight_variable([FLAGS.ecell,FLAGS.eout])
    b_1 = bias_variable([FLAGS.eout])
    X_1_fc = tf.matmul(X_1, w_1) + b_1

    w_2 = weight_variable([FLAGS.ecell,FLAGS.eout])
    b_2 = bias_variable([FLAGS.eout])
    X_2_fc = tf.matmul(X_2, w_2) + b_2


    R_A = tf.reshape(X[:, -6], [FLAGS.ebatch, 1])/23
    R_B = tf.reshape(X[:, -3], [FLAGS.ebatch, 1])/23
    logits = R_A - R_B - (X_1_fc - X_2_fc)
    entropy = tf.square(tf.ones([FLAGS.ebatch, 1]) - Y - logits)

    # logits = X_1_fc - X_2_fc
    # entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)

    loss = tf.reduce_mean(entropy)
    # tf.add_to_collection("losses",loss)
    # loss_batches = tf.reduce_mean(entropy,1)
    # loss = tf.add_n(tf.get_collection('losses'))
    optimizer = tf.train.AdamOptimizer(FLAGS.lrate).minimize(loss)


    length_0 = tf.abs(tf.zeros([FLAGS.ebatch, 1])-logits)
    length_1 = tf.abs(tf.ones([FLAGS.ebatch, 1])-logits)
    logits = tf.concat([length_1,length_0], 1)
    logits_ = tf.reshape(tf.cast(tf.argmax(logits, 1), tf.float32), [FLAGS.ebatch, 1])
    acc = tf.equal(tf.ones([FLAGS.ebatch, 1]) -Y, logits_)
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))*100 # cast：转化格式

    # y_pred = tf.arg_max(logits, 1)
    # bool_pred = tf.equal(tf.arg_max(Y, 1), y_pred)
    # accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32)) * 100  # cast：转化格式

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
            FLAGS.dropout_rate = 1.0
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, model_dir)
            total_acc = 0.0
            FLAGS.ebatch = len(all[0])
            X_batch, Y_batch = dota.test.next_batch(FLAGS.ebatch,shuffle=False)
            lgts = [0]*len(all[0])
            maxs = [0]*len(all[0])
            for i in range(10):
                accbatch, lgt,logs = sess.run([accuracy,logits,logits_],
                          feed_dict={X: X_batch,Y:Y_batch, keep_prob: 1.0})
                for j in range(len(all[0])):
                    lgts[j]+=1 - logs[j][0]
                    maxs[j]+=np.max(lgt[j])
                print(logs)
            for i in range(len(all[0])):
                print(all[1][i],lgts[i]/10,maxs[i]/10)


train()
# predict()