import time
from data_set import Reader
import tensorflow as tf
from config import *

FLAGS = tf.app.flags.FLAGS


mode = 'train'
model_dir = './checkpoint/model.ckpt-1999'

dota = Reader.read_all \
    (FLAGS.datadir, train_num=FLAGS.train_length, test_num=FLAGS.test_length, validation_num=FLAGS.Vali_length)

with tf.name_scope('data'):
    X = tf.placeholder(dtype=tf.float32, shape=(FLAGS.ebatch,FLAGS.esize*FLAGS.estep*2+6), name="inputs")
    Y = tf.placeholder(dtype=tf.float32, shape=(FLAGS.ebatch,FLAGS.eout), name='label')

keep_prob = tf.placeholder(tf.float32, name='dropout')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

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

def fc_rnn_1(X):
    input_1_1 = X[:, :, 0:FLAGS.esize-1]
    input_1_1 = tf.reshape(input_1_1, shape=[-1,FLAGS.esize-1])
    input_2_1 = X[:, :, FLAGS.esize-1]
    input_2_1 = tf.reshape(input_2_1, shape=[-1,FLAGS.estep,1])
    w_1 = weight_variable([FLAGS.esize-1,FLAGS.ecell-1])
    b_1 = bias_variable([FLAGS.ecell-1])
    X_fc_1= tf.matmul(input_1_1,w_1)+b_1
    # X_fc = tf.contrib.layers.batch_norm(X_fc, scale=True, is_training=True, updates_collections=None)
    X_fc_1 = tf.nn.dropout(X_fc_1, keep_prob)
    X_fc_1 = tf.reshape(X_fc_1,shape=[-1,FLAGS.estep,FLAGS.ecell-1])
    X_fc_1 = tf.concat([X_fc_1, input_2_1], 2)
    X_fc_1 = tf.contrib.layers.batch_norm(X_fc_1, scale=True, is_training=True, updates_collections=None)
    rnn_inputs_1 = tf.reshape(X_fc_1, (-1, FLAGS.estep, FLAGS.ecell))
    if FLAGS.rnn == 'LSTM':
        cell_1 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_1 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_1 = tf.nn.rnn_cell.DropoutWrapper \
        (cell_1, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
    init_state_1 = cell_1.zero_state(FLAGS.ebatch, dtype=tf.float32)
    rnn_outputs_1, _ = tf.nn.dynamic_rnn(cell_1, rnn_inputs_1, initial_state=init_state_1)
    return tf.transpose(rnn_outputs_1, [1, 0, 2])[-1]

def fc_rnn_2(X):
    input_1_2 = X[:, :, 0:FLAGS.esize-1]
    input_1_2 = tf.reshape(input_1_2, shape=[-1,FLAGS.esize-1])
    input_2_2 = X[:, :, FLAGS.esize-1]
    input_2_2 = tf.reshape(input_2_2, shape=[-1,FLAGS.estep,1])
    w_2 = weight_variable([FLAGS.esize-1,FLAGS.ecell-1])
    b_2 = bias_variable([FLAGS.ecell-1])
    X_fc_2 = tf.matmul(input_1_2,w_2)+b_2
    # X_fc = tf.contrib.layers.batch_norm(X_fc, scale=True, is_training=True, updates_collections=None)
    X_fc_2 = tf.nn.dropout(X_fc_2, keep_prob)
    X_fc_2 = tf.reshape(X_fc_2,shape=[-1,FLAGS.estep,FLAGS.ecell-1])
    X_fc_2 = tf.concat([X_fc_2, input_2_2], 2)
    X_fc_2 = tf.contrib.layers.batch_norm(X_fc_2, scale=True, is_training=True, updates_collections=None)
    rnn_inputs_2 = tf.reshape(X_fc_2, (-1, FLAGS.estep, FLAGS.ecell))
    if FLAGS.rnn == 'LSTM':
        cell_2 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_2 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_2 = tf.nn.rnn_cell.DropoutWrapper \
            (cell_2, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
    init_state_2 = cell_2.zero_state(FLAGS.ebatch, dtype=tf.float32)
    rnn_outputs_2, _ = tf.nn.dynamic_rnn(cell_2, rnn_inputs_2, initial_state=init_state_2,scope='2')
    return tf.transpose(rnn_outputs_2, [1, 0, 2])[-1]

def fc(X):
    input_1 = X[:, :, 0:FLAGS.esize - 1]
    input_1 = tf.reshape(input_1, shape=[-1, FLAGS.esize - 1])
    input_2 = X[:, :, FLAGS.esize - 1]
    input_2 = tf.reshape(input_2, shape=[-1, FLAGS.estep, 1])
    w = weight_variable([FLAGS.esize - 1, FLAGS.ecell - 1])
    b = bias_variable([FLAGS.ecell - 1])
    X_fc = tf.matmul(input_1, w) + b
    # X_fc = tf.contrib.layers.batch_norm(X_fc, scale=True, is_training=True, updates_collections=None)
    X_fc  = tf.nn.dropout(X_fc, keep_prob)
    X_fc = tf.reshape(X_fc, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
    X_fc = tf.concat([X_fc, input_2], 2)
    X_fc = tf.contrib.layers.batch_norm(X_fc, scale=True, is_training=True, updates_collections=None)
    return X_fc

def rnn_1_(X):
    rnn_inputs_1 = tf.reshape(X, (-1, FLAGS.estep, FLAGS.ecell))
    if FLAGS.rnn == 'LSTM':
        cell_1 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_1 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_1 = tf.nn.rnn_cell.DropoutWrapper \
        (cell_1, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
    init_state_1 = cell_1.zero_state(FLAGS.ebatch, dtype=tf.float32)
    rnn_outputs_1, _ = tf.nn.dynamic_rnn(cell_1, rnn_inputs_1, initial_state=init_state_1)
    return tf.transpose(rnn_outputs_1, [1, 0, 2])[-1]

def rnn_2_(X):
    rnn_inputs_2 = tf.reshape(X, (-1, FLAGS.estep, FLAGS.ecell))
    if FLAGS.rnn == 'LSTM':
        cell_2 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
    else:
        cell_2 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
    cell_2 = tf.nn.rnn_cell.DropoutWrapper \
            (cell_2, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
    init_state_2 = cell_2.zero_state(FLAGS.ebatch, dtype=tf.float32)
    rnn_outputs_2, _ = tf.nn.dynamic_rnn(cell_2, rnn_inputs_2, initial_state=init_state_2,scope='2')
    return tf.transpose(rnn_outputs_2, [1, 0, 2])[-1]

X_1_ = tf.reshape(X[:, 0:FLAGS.estep * FLAGS.esize], [-1, FLAGS.estep, FLAGS.esize])
X_2_ = tf.reshape(X[:, FLAGS.estep * FLAGS.esize:FLAGS.estep * FLAGS.esize * 2],
                 [-1, FLAGS.estep, FLAGS.esize])

# X_1 = (rnn_1(fc(X_1)))
# X_2 = (rnn_2(fc(X_2)))
# X_1 = (fc_rnn_1(X_1_))
# X_2 = (fc_rnn_2(X_2_))

input_1 = tf.reshape(X_1_[:, :, 0:FLAGS.esize - 1], shape=[-1, FLAGS.esize - 1])
input_1_ext = tf.reshape(X_1_[:, :, FLAGS.esize - 1], shape=[-1, FLAGS.estep, 1])
input_2 = tf.reshape(X_2_[:, :, 0:FLAGS.esize - 1], shape=[-1, FLAGS.esize - 1])
input_2_ext = tf.reshape(X_2_[:, :, FLAGS.esize - 1], shape=[-1, FLAGS.estep, 1])

fc_1_w = weight_variable([FLAGS.esize-1,FLAGS.ecell-1])
fc_1_b = bias_variable([FLAGS.ecell-1])
rnn_1 =tf.matmul(input_1,fc_1_w)+fc_1_b
# rnn_1 = tf.contrib.layers.batch_norm(rnn_1, scale=True, is_training=True, updates_collections=None)
rnn_1 = tf.nn.relu(rnn_1)
rnn_1 = tf.nn.dropout(rnn_1, keep_prob)
rnn_1 = tf.reshape(rnn_1, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
rnn_1 = tf.concat([rnn_1, input_1_ext], 2)
rnn_1 = tf.reshape(rnn_1, (-1, FLAGS.estep, FLAGS.ecell))
if FLAGS.rnn == 'LSTM':
    cell_1 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
else:
    cell_1 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
cell_1 = tf.nn.rnn_cell.DropoutWrapper \
    (cell_1, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
init_state_1 = cell_1.zero_state(FLAGS.ebatch, dtype=tf.float32)
X_1, _ = tf.nn.dynamic_rnn(cell_1, rnn_1, initial_state=init_state_1, scope='1')

fc_2_w = weight_variable([FLAGS.esize-1,FLAGS.ecell-1])
fc_2_b = bias_variable([FLAGS.ecell-1])
rnn_2 =tf.matmul(input_2,fc_2_w)+fc_2_b
# rnn_2 = tf.contrib.layers.batch_norm(rnn_2, scale=True, is_training=True, updates_collections=None)
rnn_2 = tf.nn.relu(rnn_2)
rnn_2 = tf.nn.dropout(rnn_2, keep_prob)
rnn_2 = tf.reshape(rnn_2, shape=[-1, FLAGS.estep, FLAGS.ecell - 1])
rnn_2 = tf.concat([rnn_2, input_2_ext], 2)
rnn_2 = tf.reshape(rnn_2, (-1, FLAGS.estep, FLAGS.ecell))
if FLAGS.rnn == 'LSTM':
    cell_2 = tf.contrib.rnn.LSTMCell(FLAGS.ecell)
else:
    cell_2 = tf.contrib.rnn.GRUCell(FLAGS.ecell)
cell_2 = tf.nn.rnn_cell.DropoutWrapper \
    (cell_2, output_keep_prob=keep_prob, input_keep_prob=keep_prob)  # drop_out
init_state_2 = cell_2.zero_state(FLAGS.ebatch, dtype=tf.float32)
X_2, _ = tf.nn.dynamic_rnn(cell_2, rnn_2, initial_state=init_state_2, scope='2')


X_1 = tf.transpose(X_1, [1, 0, 2])[-1]
X_2 = tf.transpose(X_2, [1, 0, 2])[-1]
X_12 = tf.concat([X_1,X_2],1)


# w = weight_variable([2*FLAGS.ecell,FLAGS.eout])
w = weight_variable([FLAGS.ecell*2,FLAGS.eout])
b = bias_variable([FLAGS.eout])
X_fc = tf.matmul(X_12, w) + b

# logits = tf.concat([X_fc, tf.ones([FLAGS.ebatch, 1]) - X_fc], 1)
# y_ = tf.concat([Y, tf.ones([FLAGS.ebatch, 1]) - Y], 1)
logits = X_fc

# entropy = tf.square(Y-logits)
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)
loss = tf.reduce_mean(entropy)
# loss_batches = tf.reduce_mean(entropy,1)

acc = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1)) #训练logits和Y 每行取最大 然后作比较
accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))*100 # cast：转化格式

optimizer = tf.train.AdamOptimizer(FLAGS.lrate).minimize(loss)

with tf.Session() as sess:
        if mode == 'train':
            max_epoch = FLAGS.epoch + 1
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            vs = tf.trainable_variables()
            print('There are %d train_able_variables in the Graph: ' % len(vs))
            for v in vs:
                print(v)
        else:
            max_epoch = 1
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, model_dir)
        start_time = time.time()
        batch_idxs_train = FLAGS.train_length // FLAGS.ebatch
        batch_idxs_vali =   FLAGS.Vali_length // FLAGS.ebatch
        batch_idxs_test = FLAGS.test_length // FLAGS.ebatch

        for epoch in range(1,max_epoch+1):
            total_loss = 0.0
            total_acc = 0.0
            for index in range(1,batch_idxs_train+1):
                X_batch, Y_batch = dota.train.next_batch(FLAGS.ebatch)
                _, accbatch,loss_batch,tst = sess.run([optimizer,accuracy, loss,X_1_],
                                                  feed_dict={X: X_batch, Y: Y_batch, keep_prob: FLAGS.dropout_rate})
                total_loss += loss_batch
                total_acc += accbatch
            print('Batch loss at epoch {}: {:5.4f}, accuracy:{:5.4f}'
                  .format(epoch,total_loss/batch_idxs_train,total_acc/batch_idxs_train))
            #validation
            if (epoch) % (2) == 0 or mode != 'train':
                total_loss = 0.0
                total_acc = 0.0
                for index in range(1, batch_idxs_vali + 1):
                    X_batch, Y_batch = dota.validation.next_batch(FLAGS.ebatch)
                    _, loss_batch,accbatch = sess.run([optimizer, loss,accuracy],
                                                  feed_dict={X: X_batch, Y: Y_batch,keep_prob: 1.0})
                    total_loss += loss_batch
                    total_acc += accbatch
                print('Validation loss at step {}:{:5.4f}, accuracy:{:5.4f}'
                      .format(epoch,total_loss/batch_idxs_vali,total_acc/batch_idxs_vali))
            if mode == 'train':
                saver.save(sess, './checkpoint/model.ckpt', epoch)
        print("Optimization Finished!")  # should be around 0.35 after 25 epochs
        print("Total time: {0} seconds".format(time.time() - start_time))
        total_loss = 0.0
        total_acc = 0.0
        for i in range(batch_idxs_test):
            X_batch, Y_batch = dota.test.next_batch(FLAGS.ebatch)
            loss_batch,accbatch = sess.run([loss,accuracy],
                                                   feed_dict={X: X_batch, Y: Y_batch, keep_prob: 1.0})
            total_loss += loss_batch
            total_acc += accbatch
        print("TEST Loss{:5.4f}, accuracy:{:5.4f}"
              .format(total_loss/batch_idxs_test,total_acc/batch_idxs_test))

