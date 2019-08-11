import tensorflow as tf
from dataset.Reader import Reader
from network.network import Network
from config import *

FLAGS = tf.app.flags.FLAGS

class Learning:
    def __init__(self):

        self.logs_dir = FLAGS.logdir
        self.data_dir = FLAGS.datadir
        self.train_logs_path = self.logs_dir + '/train_logs'
        self.test_logs_path = self.logs_dir + '/test_logs'
        self.chkpt_file = self.logs_dir + "/model.ckpt"

        self._learn_step()

    def _learn_step(self,run_options=None):

        self.is_training = True
        self.net = Network(self.is_training)
        self.train_writer = tf.summary.FileWriter(self.train_logs_path, graph=self.net.graph)
        # self.test_writer = tf.summary.FileWriter(self.test_logs_path, graph=self.net.graph)


        data_set, label_set = Reader(self.data_dir).data_read
        train_X, train_Y = data_set[0:FLAGS.train_length], label_set[0:FLAGS.train_length]
        self.test_X, self.test_Y = data_set[FLAGS.train_length:], label_set[FLAGS.train_length:]
        # train_Y_ = np.ones(len(train_Y)) - train_Y
        # test_Y_ = np.ones(len(test_Y)) - test_Y

        batch_idxs_train = FLAGS.train_length // FLAGS.ebatch
        self.batch_idxs_test = (len(data_set) - FLAGS.train_length) // FLAGS.ebatch

        max_acc_avg = 0

        #min = float(FLAGS.learning_rate_min)
        #max = float(FLAGS.learning_rate_max)

        counter = 0
        # graphs = tf.get_default_graph()
        graph = self.net.graph
        # graph.as_default()

        with tf.Session(graph = graph) as sess:
        # with tf.Session(graph=self.net.graph) as sess:

           #self._restore_checkpoint_or_init(sess)
            sess.run(tf.global_variables_initializer())
            print('Parameters were initialized')
            self.net.print_model()

            for epoch in range(1,FLAGS.epoch+1):
                learning_rate = FLAGS.lrate
                #start_time = time.time()  # 先记录这一步的时间
                #counter_ = counter
                # 训练
                train_acc_avg = 0
                if epoch%2 == 0 :
                    for idx in range(0, batch_idxs_train):
                        feed_dict = self.feed_data(train_X, train_Y, idx, FLAGS.dropout_rate, learning_rate)
                        _, summary, train_acc, train_step = sess.run(
                            [self.net.global_step, self.net.summary_op, self.net.accuracy, self.net.train_step],
                            feed_dict=feed_dict,
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=tf.RunMetadata())
                        self.train_writer.add_summary(summary, counter)
                        counter += 1
                        train_acc_avg += train_acc
                        if idx % 10 == 0:
                            print('Epoch', epoch, ',', idx * 100 // batch_idxs_train, '%')
                            # duration = time.time() - start_time
                    save_path = self.net.saver.save(sess, self.chkpt_file)
                    print('Epoch', epoch, ',', "Model saved in file: %s" % save_path)
                    if epoch%4 == 0 :
                        self._evaluate_test()
                        # self.net.saver.restore(sess, self.chkpt_file)
                else:
                    self.net.print_model()
                    for idx in range(0, batch_idxs_train):
                        feed_dict = self.feed_data(train_X, train_Y, idx, FLAGS.dropout_rate, learning_rate)

                        _, summary, train_acc, train_step = sess.run(
                             [self.net.global_step, self.net.summary_op, self.net.accuracy, self.net.train_step],
                            feed_dict=feed_dict)
                        self.train_writer.add_summary(summary, counter)
                        counter += 1
                        train_acc_avg += train_acc
                        if idx % 10 == 0:
                            print('Epoch', epoch, ',', idx * 100 // batch_idxs_train, '%')
                            # duration = time.time() - start_time
                print('Epoch %d : Train Accuracy = %.2f ;lr=%.8f' %
                              (epoch,100* train_acc_avg/batch_idxs_train,learning_rate))

                # 测试
                # test_acc_avg = 0
                # self.net.is_training = False
                # for idx in range(0, batch_idxs_test):
                #     feed_dict = self.feed_data(test_X, test_Y, idx, 1.0,learning_rate)
                #     _,summary,test_acc_,train_step = sess.run(
                #         [self.net.train_step, self.net.summary_op, self.net.accuracy,self.net.train_step],
                #         feed_dict=feed_dict, options=run_options)
                #     self.test_writer.add_summary(summary, counter_)
                #     counter_ += batch_idxs_train//batch_idxs_test
                #     #平均和最优
                #     test_acc_avg+=test_acc_
                # test_acc_avg /= batch_idxs_test
                # if test_acc_avg > max_acc_avg:
                #     max_acc_avg = test_acc_avg
                # #epoch_duration = time.time() - start_time
                # print('Epoch %d done:    Test Acccuracy (avg) = %.2f , Top (avg) = %.2f '%
                #       (epoch, 100 * test_acc_avg, 100*max_acc_avg))
                # self.net.is_training = True


    def feed_data(self,data,label,idx,keep_prob,learning_rate):
        batch_start = idx * FLAGS.ebatch
        batch_end = min(FLAGS.ebatch + batch_start, FLAGS.train_length)
        if keep_prob != 1.0:
            is_training=True,
        else:
            is_training=False,
        return  {
            self.net.x: data[batch_start:batch_end],
            self.net.y: label[batch_start:batch_end],
            self.net.keep_prob: keep_prob,
            # self.net.is_training: is_training,
        #self.learning_rate: learning_rate
        }

    def _restore_checkpoint_or_init(self, sess):
        import os
        if os.path.exists(self.chkpt_file+'.index'):
        # if os.path.exists('network/logs'):
            self.net.saver.restore(sess, self.chkpt_file)
            print("Model restored.")
        else:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            print('Parameters were initialized')
        #self.net.print_model()

    def _evaluate_train(self):
        self.keep_prob = 0.75
        self.is_training = True
        self.net = Network(self.is_training)
        self.ten_accuracy = []
        self.epoch_accuracy = []

        with tf.Session(graph=self.net.graph) as sess:
            self._restore_checkpoint_or_init(sess)

            step_num = 1
            max_steps = FLAGS.epoch * 100
            while step_num <= max_steps:
                if step_num % 10 == 0:
                    gs, acc = self._train_step(sess,
                                               tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                               tf.RunMetadata())
                    self._add_accuracy(step_num, gs, acc)
                    save_path = self.net.saver.save(sess, self.chkpt_file)
                    print("Model saved in file: %s" % save_path)
                    if step_num % 100 == 0:
                        self._evaluate_test()
                else:
                    gs, acc = self._train_step(sess)
                    self._add_accuracy(step_num, gs, acc)
                step_num += 1

    def _evaluate_test(self):
        self.keep_prob = 1.0
        self.is_training = False
        self.net = Network(self.is_training)
        self.ten_accuracy = []
        self.epoch_accuracy = []
        self.test_writer = tf.summary.FileWriter(self.test_logs_path, graph=self.net.graph)
        # global graph
        graph = self.net.graph

        with tf.Session(graph=graph) as sess:
            # self._restore_checkpoint_or_init(sess)
            self.net.saver.restore(sess, self.chkpt_file)
            # 测试
            counter_ = 0
            test_acc_avg = 0

            for idx in range(0, self.batch_idxs_test):
                feed_dict = self.feed_data(self.test_X, self.test_Y, idx, 1.0,FLAGS.lrate)
                global_step,summary,test_acc_,train_step = sess.run(
                    [self.net.global_step, self.net.summary_op, self.net.accuracy,self.net.train_step],
                        feed_dict=feed_dict, options=None)
                self.test_writer.add_summary(summary, global_step)
                counter_ += 1
                #     #平均和最优
                test_acc_avg+=test_acc_
            test_acc_avg /= self.batch_idxs_test

                 #epoch_duration = time.time() - start_time
            print('Test %d done:    Test Acccuracy (avg) = %.2f'%
                      (counter_*5, 100 * test_acc_avg,))
            # time.sleep(10)



