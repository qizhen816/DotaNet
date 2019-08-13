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
from tensorflow.contrib import rnn
from Dota.ops import *

class DotaModel(object):
    def __init__(self, sess,train_length,layer_units,net,
         input_length=22000,n_hidden = 30,
         batch_size=5, output_length=20,
         Drop_out = True,Batch_norm = True ,mix_cnn = False,
         checkpoint_dir=None, out_type = 'softmax' ,trainer = 'Adam',
         data_dir = None):
      self.sess = sess

      self.net = net
      self.mix_cnn = mix_cnn
      self.batch_size = batch_size
      self.input_length = input_length
      self.output_length = output_length
      self.train_length = train_length

      self.layer_units = layer_units.split(',')
      self.out_type = out_type
      self.trainer = trainer

      self.n_hidden = n_hidden
      self.Batch_norm = Batch_norm
      self.Drop_out = Drop_out
      self.netGraph = []

      self.d_bn1 = batch_norm(name='d_bn01')
      self.d_bn2 = batch_norm(name='d_bn02')
      self.d_bn3 = batch_norm(name='d_bn03')
      self.r_bn0 = batch_norm(name='r_bn0')
      self.d_bn_last = batch_norm(name='d_bn_last')

      #if self.Drop_out == True or self.Batch_norm == True:
      self.Batch_Norm = []
      self.Drop_Out = []
      for n in range(len(self.layer_units)):
            if self.layer_units[n][-1] == 'd':
                self.netGraph.append('  ->'+str(self.layer_units[n][0:-1])+'<-')
            else:
                self.netGraph.append('  ->' + str(self.layer_units[n]) + '<-')
            if self.Batch_norm == True:
                self.Batch_Norm.append(batch_norm(name = 'bn'+str(n)))
                self.netGraph.append('->Batch norm<-')
            if self.layer_units[n][-1] == 'd':
                if self.Drop_out == True:
                    self.Drop_Out.append(1)
                    self.netGraph.append(' ->Drop out<-')
                else:
                    self.Drop_Out.append(0)
                self.layer_units[n] = self.layer_units[n][0:-1]
            elif self.Drop_out == True:
                self.Drop_Out.append(0)
            self.layer_units[n] = int(self.layer_units[n])

      self.checkpoint_dir = checkpoint_dir
      self.data_dir = data_dir
      self.build_model()

    def feed_data(self,data,label,idx,keep_prob,learning_rate):
        batch_start = idx * self.batch_size
        batch_end = min(self.batch_size + batch_start, self.train_length)
        return  {
            self.X: data[batch_start:batch_end],
            self.Y: label[batch_start:batch_end],
            self.keep_prob: keep_prob,
            self.learning_rate: learning_rate
        }

    def build_model(self):

        self.Y = tf.placeholder("float", [self.batch_size, self.output_length],name= 'Y')
        self.X = tf.placeholder("float", [self.batch_size, self.input_length],name= 'X')
        self.keep_prob = tf.placeholder("float")
        self.learning_rate = tf.placeholder("float")

        Xinput = self.X

        if self.net == 'dnn':
            self.logits = self.run_net(Xinput)
        if self.net == 'cnn':
            self.logits = self.run_conv(Xinput)
        if self.net == 'rnn':
            self.logits = self.run_rnn(Xinput)
        if self.net == 'mix':
            self.logits = self.run_mix(Xinput)
        if self.net == 'rnn2':
            self.Y_resu,self.logits = self.run_rnnx2(X = Xinput)
            self.a_sum = tf.summary.histogram("Y_softmax", self.Y_resu)

        #尝试减法 一维
        if self.out_type == 'softmax' and self.net != 'rnn2':
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y)
            self.loss = tf.reduce_mean(self.cross_entropy)
        if self.out_type == 'sigmoid' and self.net != 'rnn2':
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            self.loss = tf.reduce_mean(self.cross_entropy)
        if self.out_type == 'linear' and self.net != 'rnn2':
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.logits-self.Y, 2)))
        if self.net == 'rnn2':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
            #self.cross_entropy = tf.square(tf.reshape(self.logits, [-1]) - tf.reshape(self.Y, [-1]))
            #self.loss = tf.reduce_mean(self.cross_entropy)
            self.loss_sum_train = tf.summary.scalar("train_loss",self.loss)
            #self.loss_sum_test = tf.summary.scalar("test_loss", self.loss)

        #t_vars = tf.trainable_variables() 不知道干啥用哒
        #self.saver = tf.train.Saver()
        self.predict = tf.argmax(self.logits, 1)
        self.acc = tf.equal(tf.argmax(self.Y,1) ,self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(self.acc, tf.float32))
        self.acc_sum_train = tf.summary.scalar("train_accuracy", self.accuracy)


    def train(self,config):
        if self.trainer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=config.beta1,beta2=0.999,epsilon=1e-8)
        train_op = optimizer.minimize(self.loss)

        try:
            init = tf.global_variables_initializer()
        except:
            init = tf.initialize_all_variables()

        self.ab_sum = tf.summary.merge_all()
        #self.b_sum = tf.summary.merge(self.b,self.b_loss)
        # 选定可视化存储目录
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        self.sess.run(init)

        data_set, label_set = self.load_datas()
        train_X, train_Y = data_set[0:self.train_length], label_set[0:self.train_length]
        test_X, test_Y = data_set[self.train_length:], label_set[self.train_length:]
        #train_Y_ = np.ones(len(train_Y)) - train_Y
        #test_Y_ = np.ones(len(test_Y)) - test_Y
        train_Y = self.dense_to_one_hot(train_Y,self.output_length)
        test_Y = self.dense_to_one_hot(test_Y,self.output_length)

        batch_idxs_train = self.train_length//config.batch_size
        batch_idxs_test = (len(data_set)-self.train_length)//config.batch_size

        max_acc_avg = 0
        max_acc_best = 0

        min = float(config.learning_rate_min)
        max = float(config.learning_rate_max)

        #画图
        #for items in self.netGraph:
         #   print(items)
        counter = 0
        for epoch in range(0,config.epoch):
            learning_rate = min + (max - min) * math.exp(-epoch / config.epoch)
            start_time = time.time()  # 先记录这一步的时间
            # 训练
            for idx in range(0,batch_idxs_train):
                feed_dict = self.feed_data(train_X,train_Y,idx,0.5,learning_rate)
                _,train_acc,summary_str =self.sess.run([train_op,self.accuracy,self.ab_sum],feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)
                counter += 1
                if idx%10 == 0:
                    duration = time.time() - start_time
                   # writer.add_summary(result, idx)  # result是summary类型的，需要放入writer中，i步数（x轴）
                    print('Epoch %d ,Step %d: accuracy = %.2f (%.3f sec);lr=%.8f' %
                          (epoch,idx,100*train_acc, duration,learning_rate))
            # 测试
            test_acc_avg = 0
            test_acc_best = 0
            for idx in range(0, batch_idxs_test):
                feed_dict = self.feed_data(test_X, test_Y, idx, 1,learning_rate)
                _,test_acc_ = self.sess.run([self.logits,self.accuracy],feed_dict=feed_dict)
                #平均和最优
                test_acc_avg+=test_acc_
                if test_acc_> test_acc_best:
                    test_acc_best = test_acc_
            test_acc_avg /= batch_idxs_test
            if test_acc_avg > max_acc_avg:
                max_acc_avg = test_acc_avg
            if test_acc_best > max_acc_best:
                max_acc_best = test_acc_best

            epoch_duration = time.time() - start_time
            print('Epoch %d done: (%.3f sec)   Test Acccuracy (avg) = %.2f (best) = %.2f , Top (avg) = %.2f (best) = %.2f'%
                  (epoch,epoch_duration, 100 * test_acc_avg,100*test_acc_best, 100*max_acc_avg,100*max_acc_best))
        #保存？
        # self.save(config.checkpoint_dir)

    def run_conv(self,X,reuse = False):
        with tf.variable_scope("hiddenlayers") as scope:
            if reuse:
                scope.reuse_variables()
        X1 = X[:,3:1023]
        X2 = X[:,1026:2046]
        x = tf.concat([X1,X2],1)
        X3 = X[:,0:3]
        X4 = X[:,1023:1026]

        x = tf.reshape(x, shape=[-1, 20, 102, 1])#输入尺寸
        # Convolution Layer
        conv1 = self.conv2d(x, tf.Variable(tf.random_normal([1, 4, 1, 32]))
                            , tf.Variable(tf.random_normal([32])))
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k1=2, k2=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, tf.Variable(tf.random_normal([2, 2, 32, 64]))
                            , tf.Variable(tf.random_normal([64])))
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k1=2, k2=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input 加上个队伍信息
        X = tf.reshape(conv2, [-1, tf.Variable(tf.random_normal([5 * 26 * 64, 1024])).get_shape().as_list()[0]])
        X = tf.concat([X4,X],1)
        X = tf.concat([X3,X],1)
        layer = []
        weights = []
        biases = []
        weights.append(tf.Variable(
            tf.truncated_normal([20, self.layer_units[0]],
                                stddev=1.0 / math.sqrt(20)), name='weights0'))  # 并非Xaiver
        biases.append(tf.Variable(tf.zeros([self.layer_units[0]]),
                                  name='biases0'))
        layer.append(tf.nn.relu(tf.matmul(X, weights[0]) + biases[0]))
        if self.Batch_norm == True:
            layer[0] = self.Batch_Norm[0](layer[0])
            # layer[0] = tf.contrib.layers.batch_norm(layer[0], scale=True, is_training=True, updates_collections=None)
        if self.Drop_out == True and self.Drop_Out[0] == 1:
            layer[0] = tf.nn.dropout(layer[0], self.keep_prob)
        # 隐藏层
        for i in range(1, len(self.layer_units)):
            weights.append(tf.Variable(
                tf.truncated_normal([self.layer_units[i - 1], self.layer_units[i]],
                                    stddev=1.0 / math.sqrt(float(self.layer_units[i - 1]))), name='weights' + str(i)))
            biases.append(tf.Variable(tf.zeros([self.layer_units[i]]),
                                      name='biases' + str(i)))
            layer.append(tf.nn.relu(tf.matmul(layer[i - 1], weights[i]) + biases[i]))
            if self.Batch_norm == True:
                layer[i] = self.Batch_Norm[i](layer[i])
                # layer[i] = tf.contrib.layers.batch_norm(layer[i], scale=True, is_training=True,
                #                                        updates_collections=None)
            if self.Drop_out == True and self.Drop_Out[i] == 1:
                layer[i] = tf.nn.dropout(layer[i], self.keep_prob)
        # 输出层
        weights.append(tf.Variable(
            tf.truncated_normal([self.layer_units[-1], self.output_length],
                                stddev=1.0 / math.sqrt(float(self.layer_units[-1]))), name='weights-1'))
        biases.append(tf.Variable(tf.zeros([self.output_length]),
                                  name='biases-1'))

        logits_fc = tf.matmul(layer[-1], weights[-1]) + biases[-1]
        # Output, class prediction

        if self.out_type == 'softmax' or self.out_type == 'linear':
            return logits_fc
        elif self.out_type == 'sigmoid':
            return tf.nn.sigmoid(logits_fc)

    def run_rnn(self,X,isOnly = True,n_hidden = 0,reuse = False,scopename = "RNN_layers"):
        with tf.variable_scope(scopename) as scope:
            if reuse:
                scope.reuse_variables()
            if isOnly == True:
                n_hidden = self.n_hidden
                X1 = X[:,0:1020]
                X2 = X[:,1020:2040]
                x = tf.concat([X1,X2],1)
                x = tf.reshape(x, [-1, 20, 102])
                X1 = x[:, :, 0:50]
                X2 = x[:, :, 51:101]
                X_RNN = tf.concat([X1, X2], 2)
                X_RNN = tf.reshape(X_RNN,[-1,100])
                X_RNN = tf.matmul(X_RNN, tf.get_variable('PreRNN_w',[100, n_hidden],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(100.0))))\
                        + tf.get_variable('PewRNN_b',[n_hidden],initializer=tf.constant_initializer(0.0))
                tf.summary.histogram("PreRNN_wx+b", X_RNN)
                X_RNN= tf.contrib.layers.batch_norm(X_RNN, scale=True, is_training=True, updates_collections=None)
                X_RNN = tf.reshape(X_RNN,[-1,20,n_hidden])
            else:
                X_RNN = X
                tf.summary.histogram("PreRNN_wx+b", X_RNN)

            lstm_cell = rnn.BasicLSTMCell(n_hidden)
                # lstm_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
                # lstm_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
            if self.Drop_out == True:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)  # drop_out

            init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_RNN, initial_state=init_state, dtype=tf.float32)
            output = tf.transpose(outputs, [1, 0, 2])[-1]

            rnn_out = tf.matmul(output,tf.get_variable('OutRNN_w',[n_hidden, self.output_length],
                    initializer=tf.contrib.layers.xavier_initializer()))\
                      + tf.get_variable('OutRNN_b', [self.output_length], initializer=tf.constant_initializer(0.0))
            #math.sqrt(2.0 / n_hidden))))\      1.8 / math.sqrt(n_hidden)))) \
            #tf.summary.histogram("RNN_out", rnn_out)
            return rnn_out

    def run_mix(self, X, isOnly = True,reuse = False,scopename = "Mix_layers"):
        with tf.variable_scope(scopename) as scope:
            if reuse:
                scope.reuse_variables()
            n_hidden = self.n_hidden
            n_steps = 20
            n_data = 100

            if isOnly == True:
                X1 = X[:,0:1020]
                X2 = X[:,1020:2040]
                #X3 = X[:,2040:2043]
                #X4 = X[:,2043:2046]
                x = tf.concat([X1,X2],1)
                x = tf.reshape(x,[-1,20,102])
                X1 = x[:,:,0:50]
                X2 = x[:,:,51:101]

                X_RNN = tf.concat([X1, X2], 2)

                X_result1 = x[:,:,50]
                X_result2 = x[:,:,101]
                X_results = tf.concat([X_result1,X_result2],1)
                X_results = tf.reshape(X_results,
                            [-1, tf.Variable(tf.random_normal([40, self.layer_units[0]])).get_shape().as_list()[0]])
            else:
                n_data = 50
                X_RNN = X[:,:,0:50]
                X_results = X[:,:,50]

            #20场比赛结果的感知机部分：
            #logit_fc = self.run_net(X = X_results,isOnly = False,reuse = reuse)
            #self.r_bn1 = batch_norm(name='DNN_bn_1')
            #logit_fc = self.r_bn1(logit_fc)
            #logit_fc = tf.nn.relu(logit_fc)
            #tf.summary.histogram("DNN_out", logit_fc)

            #是否在RNN之前CNN： 使用效果很棒
            if self.mix_cnn == True:
                n_conv1 = 2
                n_conv2 = 4
                n_conv3 = n_hidden

                x = tf.reshape(X_RNN, shape=[-1, n_steps,n_data,1])  # 输入尺寸
                    # Convolution Layer
                conv1 = conv2d_rnn(x,input_dim = 1,
                                   output_dim=n_conv1,k_h=1,k_w=2,d_h=1,d_w=2,name='conv1',reuse= reuse)
                conv1 = (self.d_bn1(conv1))
                conv2 = conv2d_rnn(conv1,input_dim = n_conv1,
                                   output_dim=n_conv2,k_h=1,k_w=5,d_h=1,d_w=5,name='conv2',reuse= reuse)
                conv2 = (self.d_bn2(conv2))
                conv3 = conv2d_rnn(conv2,input_dim = n_conv2,
                                   output_dim=n_conv3,k_h=1,k_w=5,d_h=1,d_w=5,name='conv3',reuse= reuse)
                tf.summary.histogram("CONV_3", conv3)
                #conv3 = lrelu(self.d_bn3(conv3))
                conv3 = (self.d_bn3(conv3)) #还可以 lrelu?
                #conv3 = self.d_bn3(conv3)
                #conv3 = tf.contrib.slim.dropout(conv3, self.keep_prob, scope='Dropout_3') #测试中 结果似乎不好
                #conv3 = linear(tf.reshape(conv3, [-1,n_conv3]),n_hidden,scope = 'linear',reuse = reuse) #加FC层测试效果不好
                #tf.summary.histogram("CONV_3_liner", conv3)
                #X_RNN = self.d_bn_last(conv3) #测试效果不好
                X_RNN = conv3
                #X_RNN = tf.nn.relu(X_RNN) #测试效果不好
                #X_RNN = tf.nn.dropout(X_RNN, self.keep_prob) #测试效果不好
                X_RNN = tf.reshape(X_RNN,[-1,n_steps,n_hidden])
                tf.summary.histogram("LSTM_in_CNN", X_RNN)
                X_results = tf.reshape(X_results, [5, 20, 1]) #？ 5->-1?
                X_RNN = tf.concat([X_RNN,X_results],2)
                tf.summary.histogram("LSTM_in_ALL", X_RNN)

                #X_result1 = tf.reshape(X_result1,[5,20,1])
                #X_result2 = tf.reshape(X_result2,[5,20,1])
                #X_RNN = tf.concat([X_RNN,X_result1],2)
                #X_RNN = tf.concat([X_RNN,X_result2],2)
            else:
                X_RNN = tf.reshape(X_RNN, [-1, n_data])
                X_RNN = tf.matmul(X_RNN, tf.get_variable('InRNN_w',[n_data, n_hidden],
                         initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(n_data)))) \
                        + tf.get_variable('InRNN_b',[n_hidden],initializer=tf.constant_initializer(0.0))
                X_RNN = tf.nn.relu(X_RNN)
                X_RNN = tf.contrib.layers.batch_norm(X_RNN, scale=True, is_training=True, updates_collections=None)
                X_RNN = tf.reshape(X_RNN, [-1, n_steps, n_hidden])
                tf.summary.histogram("LSTM_in", X_RNN)

            '''
            #x = tf.unstack(X_RNN, n_steps, 1) #静态RNN
    
            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(n_hidden)
            #lstm_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
            #lstm_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)#drop_out
    
            init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
    
            #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_RNN, initial_state=init_state, dtype=tf.float32)
    
            output = tf.transpose(outputs, [1, 0, 2])[-1]
    
            #rnn_out = tf.matmul(outputs[-1], tf.Variable(tf.random_normal([n_hidden, 2]))) + tf.Variable(tf.random_normal([2]))
            rnn_out = tf.matmul(output, tf.Variable(tf.truncated_normal([n_hidden, self.output_length],
                                    stddev= 1.0 / n_hidden))) + tf.Variable(tf.zeros([self.output_length]))
    
            #rnn_out = tf.nn.relu(rnn_out)
            '''
            #RNN部分：
            rnn_out = self.run_rnn(X_RNN,isOnly=False,n_hidden = n_hidden+1,reuse = reuse)
            tf.summary.histogram("RNN_out", rnn_out)
            #                                                                                                                                           rnn_out = tf.nn.dropout(rnn_out, self.keep_prob) #没测试

            #self.r_bn = batch_norm(name='r_bn0') #？
            #rnn_out = self.r_bn0(rnn_out)
            #rnn_out = lrelu(rnn_out)

            #out = tf.matmul(logit_fc, tf.get_variable('Mixout_w_dnn',[self.output_length, self.output_length],
             #               initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(self.output_length)))) \
            #out = tf.matmul(rnn_out, tf.get_variable('Mixout_w_rnn',[self.output_length, self.output_length],
             #        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(self.output_length)))) \
              #    + tf.get_variable('Mixout_b', [self.output_length], initializer=tf.constant_initializer(0.0))

            #tf.summary.histogram("Mix_out", out)

            return rnn_out

    def run_rnnx2(self,X,reuse = False,scopename = "2RNN_layers"):
        with tf.variable_scope(scopename) as scope:
            if reuse:
                scope.reuse_variables()

            X1 = tf.reshape(X[:,0:1020],[-1,20,51])
            X2 = tf.reshape(X[:,1020:2040],[-1,20,51])

            A = X[:,2040:2043]
            B = X[:,2043:2046]
            #tf.nn.relu 不用relu效果好
            a_out = (self.run_mix(X=X1,isOnly=False,scopename="sn_a"))
            b_out = (self.run_mix(X=X2,isOnly=False,reuse = False,scopename="sn_b"))

            out_ = tf.concat([a_out,b_out],1)
            #out_ = tf.concat([A,out_],1)
            #B = B - A
            #out_ = tf.concat([out_,B],1)

            out_w = out_.get_shape().as_list()[1]
            out = tf.matmul(out_, tf.get_variable('Final_out_w',[out_w, self.output_length],
                    initializer=tf.contrib.layers.xavier_initializer())) \
                +tf.get_variable('Final_out_b', [self.output_length], initializer=tf.constant_initializer(0.0))
            #math.sqrt(2.0 / float(out_w)))))\ 1.0 / math.sqrt(float(out_w))))) \
            tf.summary.histogram("Final_out", out)
        return tf.nn.softmax(out),out
        #return tf.nn.softmax(a_out),a_out,tf.nn.softmax(b_out),b_out

    def run_net(self,X,reuse = False,isOnly = True,scopename = "DNN_layers"):
        with tf.variable_scope(scopename) as scope:
            if reuse:
                scope.reuse_variables()

            if isOnly == False:
                weight_fc0 = tf.get_variable('DNN_w_0', [int(X.get_shape().as_list()[1]), self.layer_units[0]],
                                    initializer=tf.truncated_normal_initializer(
                                        stddev=1.0 / math.sqrt(float(int(X.get_shape().as_list()[1])))))
                biases_fc0 = tf.get_variable('DNN_b_0', [self.layer_units[0]], initializer=tf.constant_initializer(0.0))
                h_fc0 = tf.matmul(X, weight_fc0) + biases_fc0
                self.dnn_bn0 = batch_norm(name='DNN_bn_0')
                h_fc0 = self.dnn_bn0(h_fc0)
                h_fc0 = tf.nn.relu(h_fc0)
                weight_fc1 = tf.get_variable('DNN_w_-1', [self.layer_units[-1], self.output_length],
                                    initializer=tf.truncated_normal_initializer(
                                        stddev=1.0 / math.sqrt(float(self.layer_units[-1]))))
                biases_fc1 = tf.get_variable('DNN_b_-1',
                                              [self.output_length], initializer=tf.constant_initializer(0.0))
                logits = tf.matmul(h_fc0, weight_fc1) + biases_fc1
                return logits

            layer = []
            weights = []
            biases = []
            #第一层
            #weights.append(tf.get_variable('DNN_w_0',tf.truncated_normal([int(X.get_shape().as_list()[1]), self.layer_units[0]],
            #                 stddev= 1.0 / math.sqrt(float(int(X.get_shape().as_list()[1]))))))
            weights.append(
                tf.get_variable('DNN_w_0', [int(X.get_shape().as_list()[1]), self.layer_units[0]],
             initializer=tf.truncated_normal_initializer(stddev= 1.0 / math.sqrt(float(int(X.get_shape().as_list()[1]))))))
            biases.append(tf.get_variable('DNN_b_0', [self.layer_units[0]],initializer=tf.constant_initializer(0.0)))
            tf.summary.histogram("DNN_in_w", weights[0])
            tf.summary.histogram("DNN_in_b", biases[0])
            layer.append((tf.matmul(X, weights[0]) + biases[0]))
            if self.Batch_norm == True:
                layer[0] = self.Batch_Norm[0](layer[0])
                #layer[0] = tf.contrib.layers.batch_norm(layer[0], scale=True, is_training=True, updates_collections=None)
                tf.summary.histogram("DNN_bn", layer[0])
            layer[0] = tf.nn.relu(layer[0])
            if self.Drop_out == True and self.Drop_Out[0] == 1:
                layer[0] = tf.nn.dropout(layer[0], self.keep_prob)
            #隐藏层
            for i in range(1,len(self.layer_units)):
                weights.append(
                    tf.get_variable('DNN_w_'+str(i),[self.layer_units[i-1], self.layer_units[i]],
                    initializer=tf.truncated_normal_initializer(stddev= 1.0 / math.sqrt(float(self.layer_units[i-1])))))
                biases.append(tf.get_variable('DNN_b_'+str(i),
                                              [self.layer_units[i]],initializer=tf.constant_initializer(0.0)))
                layer.append((tf.matmul(layer[i-1], weights[i]) + biases[i]))
                if self.Batch_norm == True:
                    layer[i] = self.Batch_Norm[i](layer[i])
                    #layer[i] = tf.contrib.layers.batch_norm(layer[i], scale=True, is_training=True,
                    #                                        updates_collections=None)
                layer[i] = tf.nn.relu(layer[i])
                if self.Drop_out == True and self.Drop_Out[i] == 1:
                    layer[i] = tf.nn.dropout(layer[i], self.keep_prob)
            #输出层/选择分类器？
            weights.append(
                    tf.get_variable('DNN_w_-1',[self.layer_units[-1], self.output_length],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.layer_units[-1])))))
            biases.append(tf.get_variable('DNN_b_-1',
                                          [self.output_length],initializer=tf.constant_initializer(0.0)))
            logits = tf.matmul(layer[-1], weights[-1]) + biases[-1]

            if self.out_type == 'softmax' or self.out_type == 'linear':
                return logits
            elif self.out_type == 'sigmoid':
                return tf.nn.sigmoid(logits)

    def load_datas(self):
        f = gzip.open(self.data_dir,'rb')
        data_set,label_set  = pickle.load(f, encoding='latin1')
        f.close()
        return data_set,label_set

    def dense_to_one_hot(self, labels_dense, num_classes=20):
        num_labels = labels_dense.shape[0]
        labels_one_hot = np.zeros((num_labels, num_classes))
        if num_classes != 1:
            for i in range(0, num_labels):
                labels_one_hot[i][int(labels_dense[i])] = 1
        else:
            for i in range(0, num_labels):
                labels_one_hot[i] = int(labels_dense[i])
        return labels_one_hot

    def save(self, checkpoint_dir, ):
        model_name = "DotaModel.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        #self.saver.save(self.sess,
           #             os.path.join(checkpoint_dir, model_name))
