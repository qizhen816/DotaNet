import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('ebatch', 5, 'Number of batch')
tf.app.flags.DEFINE_integer('esize', 51, 'Size of examples')
tf.app.flags.DEFINE_integer('eout',2 , 'Length of output')
tf.app.flags.DEFINE_integer('estep', 20, 'Length of step for rnn')
tf.app.flags.DEFINE_integer('ecell', 10, 'Cells of rnn')
tf.app.flags.DEFINE_integer('epoch', 4000, 'Steps')
tf.app.flags.DEFINE_integer('train_length',710, 'Train length')
tf.app.flags.DEFINE_integer('Vali_length', 100, 'Vali length')
tf.app.flags.DEFINE_integer('test_length', 100, 'Vali length')
#tf.app.flags.DEFINE_integer('height', 240, 'Height of frames')
#tf.app.flags.DEFINE_integer('width', 320, 'Width of frames')
tf.app.flags.DEFINE_float('lrate', 3e-4, 'Learning rate')
tf.app.flags.DEFINE_float('dropout_rate', 0.4, 'Drop out rate')
tf.app.flags.DEFINE_string('datadir', 'data_set/DotaTMzScore.set', 'Path to dataset')
tf.app.flags.DEFINE_string('logdir', 'network/logs', 'Path to store logs and checkpoints')
#tf.app.flags.DEFINE_string('conv', 'standard', 'Type of CNN block')
tf.app.flags.DEFINE_string('rnn', 'LSTM', 'Type of RNN block (LSTM/GRU)')
tf.app.flags.DEFINE_boolean('update', False, 'Generate TFRecords')
tf.app.flags.DEFINE_boolean('download', False, 'Download dataset')
tf.app.flags.DEFINE_boolean('restore', False, 'Restore from previous checkpoint')
tf.app.flags.DEFINE_boolean('test', False, 'Test evaluation')


data_dir = os.path.join('dataset/data/')