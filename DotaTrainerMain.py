'''

网络部分
双流LSTM

作者：戚朕
时间：2019年8月13日21:32:13
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DotaModel import DotaModel

import pprint
import tensorflow.contrib.slim as slim
import tensorflow as tf

# 模型的基本参数，类似C语言开头的defined
flags = tf.app.flags
flags.DEFINE_integer("epoch", 2000, "Epoch to train.")
flags.DEFINE_integer("train_length",760, "The size of train data.") #370 750
flags.DEFINE_integer('input_length', 2046, 'Length of input data.') #2046 2040
flags.DEFINE_integer('output_length', 2, 'Sum of classes.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.  ')
flags.DEFINE_integer('n_hidden',8, 'Hidden layers of LSTM.  ')
flags.DEFINE_integer('n_step', 20, 'Step of LSTM.  ')
flags.DEFINE_float('learning_rate_min', 0.0000003, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_max', 0.0003, 'Initial learning rate.') #3e-5 3e-6 1e-4 1e-5
flags.DEFINE_float('drop_out_rate', 0.6, 'Momentem of Adam.')
flags.DEFINE_float('beta1', 0.9, 'Momentem of Adam.')
flags.DEFINE_string('net', 'rnn2', 'Kind of networks, dnn,cnn,rnn,mix,rnn2.')
flags.DEFINE_string('layer_units', '400', 'Number of units in hidden layers.')
#flags.DEFINE_string('layer_units', '2000,1000d,500d,100d', 'Number of units in hidden layers.')
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string('data_dir', 'DotaTM0707zScore.set', 'Directory to put data.')
flags.DEFINE_string('out_type', 'softmax', 'Final classifier.')
flags.DEFINE_string('trainer', 'Adam', 'Adam or SGD.')
flags.DEFINE_boolean('Batch_norm', True, 'Using batch normalization.')
flags.DEFINE_boolean('Drop_out', True, 'Using batch drop out.')
flags.DEFINE_boolean('mix_cnn', True, 'Using conv layers in mixed net.')
FLAGS = flags.FLAGS

#Best results:
#67.2% 3e-4~3e-6 2-4-8 没有fc conv relu 不加dropout RNNdropout
#66.4% _64 3e-4~3e-7 alpha = 5
#68%  3e-4~3e-7 alpha = 6 偶然性 ?
#64%  3e-5~3e-7 appha = 6 初始化 0.0001 Xa Xa / 0.00001 0.00001 0.00001
#62.4% _64 1e-4~3e-7 alpha = 7
#64% _64 3e-5~3e-7 alpha = 7 1e-6 1e-5 1e-5
#64.8% _64 3e-4~3e-7 alpha = 7 1e-6 1e-5 1e-5
#64.8% _64 3e-4~3e-7 alpha = 6 1e-7 1e-6 1e-6 Drop0.6
#64.8%  1e-4~3e-7 alpha = 3 1e-6 1e-6 1e-6 Drop0.6
#67.2%  1e-4~3e-7 alpha = 3 1e-6 1e-6 1e-6 Drop0.6 x 2
#66.15% 3e-4~3e-7 alpha = 6 1e-7 1e-6 1e-6 Drop0.6 x 2 axis = 0


#学习率




def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
def main(_):
    pp= pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        DotaNet = DotaModel(
            sess=sess,
            net = FLAGS.net,
            mix_cnn = FLAGS.mix_cnn,
            n_hidden = FLAGS.n_hidden,
            n_step = FLAGS.n_step,
            drop_out_rate = FLAGS.drop_out_rate,
            train_length = FLAGS.train_length,
            layer_units = FLAGS.layer_units,
            input_length = FLAGS.input_length,
            batch_size = FLAGS.batch_size,
            output_length = FLAGS.output_length,
            Drop_out = FLAGS.Drop_out,
            Batch_norm = FLAGS.Batch_norm,
            checkpoint_dir = FLAGS.checkpoint_dir,
            out_type = FLAGS.out_type,
            trainer = FLAGS.trainer,
            data_dir = FLAGS.data_dir,
        )
    show_all_variables()
    DotaNet.train(FLAGS)


if __name__ == '__main__':
    #with tf.device('/cpu:0'):
        tf.app.run()