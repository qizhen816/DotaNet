from config import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, inputs, is_training, keep_prob):
        self.inputs = inputs
        self.is_training = is_training
        self.keep_prob = keep_prob
        # self.logits = self._init_model()

    def _init_model(self):
        with tf.name_scope('Input'):
            X_1 = tf.reshape(self.inputs[:, 0:FLAGS.estep * FLAGS.esize], [-1, FLAGS.estep, FLAGS.esize])
            X_2 = tf.reshape(self.inputs[:, FLAGS.estep * FLAGS.esize:FLAGS.estep * FLAGS.esize*2],
                            [-1, FLAGS.estep, FLAGS.esize])

        # net_1 = self._cnn(X_1,scope='1')
        net_1 = self._fc(X_1,scope='1')
        rnn_1 = self._rnn_cell(self.keep_prob,net_1,scope='1')

        # net_2 = self._cnn(X_2,scope='2')
        net_2 = self._fc(X_2,scope='2')
        rnn_2 = self._rnn_cell(self.keep_prob,net_2,scope='2')

        rnn = tf.concat([rnn_1,rnn_2],1)
        return self._dense(rnn)

    def _fc(self,X,reuse = False,scope = 'fc'):
        input_1 = X[:, :, 0:FLAGS.esize-1]
        self.input_1 = tf.reshape(input_1, shape=[-1,FLAGS.esize-1])
        input_2 = X[:, :, FLAGS.esize-1]
        input_2 = tf.reshape(input_2, shape=[-1,FLAGS.estep,1])
        with tf.variable_scope('FullyConnected'+scope, reuse=reuse):
            w = tf.get_variable('fc_w',[FLAGS.esize-1,8],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('fc_b',[8],
                                initializer=tf.constant_initializer(0.0))
            X_fc= tf.matmul(self.input_1,w)+b
            # X_fc = tf.contrib.layers.batch_norm(X_fc, scale=True, is_training=True, updates_collections=None)
            X_fc = tf.nn.dropout(X_fc, self.keep_prob)
            X_fc = tf.reshape(X_fc,shape=[-1,FLAGS.estep,8])
            X_fc = tf.concat([X_fc, input_2], 2)
            X_fc = tf.contrib.layers.batch_norm(X_fc, scale=True, is_training=True, updates_collections=None)
            self.x_fc = X_fc
            return X_fc


    def _cnn(self, X,reuse = False,scope = 'cnn'):
        input_1 = X[:, :, 0:FLAGS.esize-1]
        input_1 = tf.reshape(input_1, shape=[-1,FLAGS.estep,FLAGS.esize-1,1])
        input_2 = X[:, :, FLAGS.esize-1]
        input_2 = tf.reshape(input_2, shape=[-1,FLAGS.estep,1])
        with tf.variable_scope('Convolution'+scope, reuse=reuse):
            w1 = tf.get_variable('w1', [1, 50, 1, 8], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b1 = tf.get_variable('b1', [8], initializer=tf.constant_initializer(0.0))
            w2 = tf.get_variable('w2', [1, 5, 2, 4], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b2 = tf.get_variable('b2', [4], initializer=tf.constant_initializer(0.0))
            w3 = tf.get_variable('w3', [1, 2, 4, 8], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b3 = tf.get_variable('b3', [8], initializer=tf.constant_initializer(0.0))

            conv1 = tf.nn.conv2d(input_1,  filter=w1, strides= [1,1,1,1], padding='SAME')
            conv1 = conv1 + b1

            conv1 = tf.nn.dropout(tf.nn.relu(conv1), self.keep_prob)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 50, 1], strides=[1, 1, 50, 1], padding='SAME')
            # 1xCNN

            # conv2 = tf.nn.conv2d(conv1,  filter = w2, strides= [1,1,5,1], padding='SAME')
            # conv2 = conv2 + b2
            # conv2 = tf.nn.dropout(tf.nn.relu(conv2), self.keep_prob)
            # # pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 5, 1], strides=[1, 1, 5, 1], padding='SAME')
            #
            # conv3 = tf.nn.conv2d(conv2,  filter = w3, strides= [1,1,2,1], padding='SAME')
            # conv3 = conv3 + b3
            # conv3 = tf.nn.dropout(tf.nn.relu(conv3), self.keep_prob)
            # # pool3 = tf.nn.max_pool(conv3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


            pool3 = tf.reshape(pool1, shape=[-1, FLAGS.estep, FLAGS.ecell-1])
            # pool1 = tf.reshape(pool1, shape=[-1, FLAGS.estep, FLAGS.ecell-1])
            net = tf.concat([pool3, input_2], 2)
            return net

    def _inception_cnn(self, inputs):
        conv1 = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        conv2 = slim.conv2d(conv1, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_2a_3x3')
        inc_inputs = slim.conv2d(conv2, 64, [3, 3], scope='Conv2d_2b_3x3')
        with slim.arg_scope([slim.conv2d], trainable=self.is_training, stride=1, padding='SAME'):
            with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
                with tf.variable_scope('BlockInceptionA', [inc_inputs]):
                    with tf.variable_scope('IBranch_0'):
                        ibranch_0 = slim.conv2d(inc_inputs, 96, [1, 1], scope='IConv2d_0a_1x1')
                    with tf.variable_scope('IBranch_1'):
                        ibranch_1_conv1 = slim.conv2d(inc_inputs, 64, [1, 1], scope='IConv2d_0a_1x1')
                        ibranch_1 = slim.conv2d(ibranch_1_conv1, 96, [3, 3], scope='IConv2d_0b_3x3')
                    with tf.variable_scope('IBranch_2'):
                        ibranch_2_conv1 = slim.conv2d(inc_inputs, 64, [1, 1], scope='IConv2d_0a_1x1')
                        ibranch_2_conv2 = slim.conv2d(ibranch_2_conv1, 96, [3, 3], scope='IConv2d_0b_3x3')
                        ibranch_2 = slim.conv2d(ibranch_2_conv2, 96, [3, 3], scope='IConv2d_0c_3x3')
                    with tf.variable_scope('IBranch_3'):
                        ibranch_3_pool = slim.avg_pool2d(inc_inputs, [3, 3], scope='IAvgPool_0a_3x3')
                        ibranch_3 = slim.conv2d(ibranch_3_pool, 96, [1, 1], scope='IConv2d_0b_1x1')
                    inception = tf.concat(axis=3, values=[ibranch_0, ibranch_1, ibranch_2, ibranch_3])
                with tf.variable_scope('BlockReductionA', [inception]):
                    with tf.variable_scope('RBranch_0'):
                        rbranch_0 = slim.conv2d(inception, 384, [3, 3], stride=2, padding='VALID',
                                                scope='RConv2d_1a_3x3')
                    with tf.variable_scope('RBranch_1'):
                        rbranch_1_conv1 = slim.conv2d(inception, 192, [1, 1], scope='RConv2d_0a_1x1')
                        rbranch_1_conv2 = slim.conv2d(rbranch_1_conv1, 224, [3, 3], scope='RConv2d_0b_3x3')
                        rbranch_1 = slim.conv2d(rbranch_1_conv2, 256, [3, 3], stride=2, padding='VALID',
                                                scope='RConv2d_1a_3x3')
                    with tf.variable_scope('RBranch_2'):
                        rbranch_2 = slim.max_pool2d(inception, [3, 3], stride=2, padding='VALID',
                                                    scope='RMaxPool_1a_3x3')
                return tf.concat(axis=3, values=[rbranch_0, rbranch_1, rbranch_2])

    @staticmethod
    def _rnn_cell(keep_prob,net,reuse = False,scope = 'rnn'):
        with tf.variable_scope('RNN_cell'+scope,reuse=reuse):
            size = np.prod(net.get_shape().as_list()[1:])
            rnn_inputs = tf.reshape(net, (-1, FLAGS.estep, FLAGS.ecell))
            if FLAGS.rnn == 'LSTM':
                cell = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
            else:
                cell = tf.contrib.rnn.GRUCell(FLAGS.ecell)
            cell = tf.nn.rnn_cell.DropoutWrapper \
                (cell, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
            init_state = cell.zero_state(FLAGS.ebatch, dtype=tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            return tf.transpose(rnn_outputs, [1, 0, 2])[-1]
            # return tf.reduce_mean(rnn_outputs, axis=1)

    @staticmethod
    def _dense(output):
        with tf.name_scope('Dense'):
            # return slim.fully_connected(output, 2, scope="dense")
            w = tf.get_variable('w', [18,FLAGS.eout], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [FLAGS.eout], initializer=tf.constant_initializer(0.0))
            return tf.matmul(output,w)+b