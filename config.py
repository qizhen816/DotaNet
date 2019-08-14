import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('ebatch', 10, 'Number of batch')
tf.app.flags.DEFINE_integer('esize', 51, 'Size of examples')
tf.app.flags.DEFINE_integer('eout',1 , 'Length of output')
tf.app.flags.DEFINE_integer('estep', 20, 'Length of step for rnn')
tf.app.flags.DEFINE_integer('ecell', 20, 'Cells of rnn')
tf.app.flags.DEFINE_integer('epoch', 2000, 'Steps')
tf.app.flags.DEFINE_integer('train_length', 3450, 'Train length') #1320
tf.app.flags.DEFINE_integer('Vali_length', 300, 'Vali length') #100
tf.app.flags.DEFINE_integer('test_length', 295, 'Vali length') #95
tf.app.flags.DEFINE_float('lrate', 6e-4, 'Learning rate')
tf.app.flags.DEFINE_float('dropout_rate', 0.35, 'Drop out rate')
tf.app.flags.DEFINE_string('datadir', 'data_set/2data_190814.set', 'Path to dataset') #DotaTMzScore.set
tf.app.flags.DEFINE_string('logdir', 'network/logs', 'Path to store logs and checkpoints')
tf.app.flags.DEFINE_string('rnn', 'LSTM', 'Type of RNN block (LSTM/GRU)')
tf.app.flags.DEFINE_boolean('update', False, 'Generate TFRecords')
tf.app.flags.DEFINE_boolean('restore', False, 'Restore from previous checkpoint')
tf.app.flags.DEFINE_boolean('test', False, 'Test evaluation')

data_dir = os.path.join('dataset/data/')
# 2data.set