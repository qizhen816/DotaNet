import tensorflow as tf
from dataset import Reader
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
        if FLAGS.test == False:
            self._learn_step()
        else:
            self._test_step()

    def _learn_step(self,run_options=None):

        self.is_training = True
        self.net = Network(self.is_training)
        self.train_writer = tf.summary.FileWriter(self.train_logs_path, graph=self.net.graph)
        self.test_writer = tf.summary.FileWriter(self.test_logs_path, graph=self.net.graph)

        dota = Reader.read_all\
            (self.data_dir,train_num=FLAGS.train_length,test_num=FLAGS.test_length,validation_num=FLAGS.Vali_length)

        # train_Y_ = np.ones(len(train_Y)) - train_Y
        # test_Y_ = np.ones(len(test_Y)) - test_Y

        batch_idxs_train = FLAGS.train_length // FLAGS.ebatch
        batch_idxs_vali =   FLAGS.Vali_length // FLAGS.ebatch
        batch_idxs_test = FLAGS.test_length // FLAGS.ebatch

        max_acc_avg = 0
        max_vali_avg = 0

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
                counter_ = counter
                # 训练
                train_acc_avg = 0
                for idx in range(0, batch_idxs_train):
                    train_X,train_Y = dota.train.next_batch(FLAGS.ebatch)
                    feed_dict = self.feed_data(train_X, train_Y, idx, FLAGS.dropout_rate, learning_rate)
                    test,_,loss, summary, train_acc, train_step = sess.run(
                            [self.net.logits_,self.net.global_step,self.net.loss, self.net.summary_op, self.net.accuracy, self.net.train_step],
                            feed_dict=feed_dict,)
                    self.train_writer.add_summary(summary, counter)
                    counter += 1
                    train_acc_avg += train_acc
                    if idx % 10 == 0:
                        print('Epoch', epoch, ',', idx * 100 // batch_idxs_train, '%')
                            # duration = time.time() - start_time
                    # save_path = self.net.saver.save(sess, self.chkpt_file)
                # print('Epoch', epoch, ',', "Model saved in file: %s" % save_path)
                print('Epoch %d : Train Accuracy = %.2f ;lr=%.8f,loss=%.5f' %
                              (epoch,100* train_acc_avg/batch_idxs_train,learning_rate,loss))

                if epoch % 4 == 0:
                    #  验证
                    vali_acc_avg = 0
                    self.net.is_training = False
                    for idx in range(0, batch_idxs_vali):
                        vali_X,vali_Y = dota.validation.next_batch(FLAGS.ebatch)
                        feed_dict = self.feed_data(vali_X, vali_Y, idx, 1.0,learning_rate)
                        _,loss,summary,test_acc_,train_step = sess.run(
                            [self.net.global_step,self.net.loss, self.net.summary_op, self.net.accuracy,self.net.train_step],
                            feed_dict=feed_dict, options=run_options)
                        self.test_writer.add_summary(summary, counter_)
                        counter_ += batch_idxs_train//batch_idxs_test
                        #平均和最优
                        vali_acc_avg+=test_acc_
                    vali_acc_avg /= batch_idxs_vali
                    if vali_acc_avg > max_vali_avg:
                        max_vali_avg = vali_acc_avg
                    #epoch_duration = time.time() - start_time
                    print('Epoch %d done:    Vali Acccuracy (avg) = %.2f , Top (avg) = %.2f,loss=%.5f'%
                          (epoch, 100 * vali_acc_avg, 100*max_vali_avg,loss))
                    self.net.is_training = True
            save_path = self.net.saver.save(sess, self.chkpt_file)
            test_acc_avg = 0
            counter_  = 0
            for idx in range(0, batch_idxs_test):
                test_X, test_Y = dota.test.next_batch(FLAGS.ebatch)
                feed_dict = self.feed_data(test_X, test_Y, idx, 1.0, FLAGS.lrate)
                loss, summary, test_acc_, train_step = sess.run(
                                [self.net.loss, self.net.summary_op, self.net.accuracy,self.net.train_step],
                                feed_dict=feed_dict, options=run_options)
                # self.test_writer.add_summary(summary, counter_)
                counter_ += batch_idxs_train // batch_idxs_test
                # 平均和最优
                test_acc_avg += test_acc_
            test_acc_avg /= batch_idxs_test
            print('Test done,',test_acc_avg,loss)

    def _test_step(self,run_options=None):

        self.is_training = False
        self.net = Network(self.is_training)
        self.train_writer = tf.summary.FileWriter(self.train_logs_path, graph=self.net.graph)
        self.test_writer = tf.summary.FileWriter(self.test_logs_path, graph=self.net.graph)

        dota = Reader.read_all\
            (self.data_dir,train_num=FLAGS.train_length,test_num=FLAGS.test_length,validation_num=FLAGS.Vali_length)
        # train_Y_ = np.ones(len(train_Y)) - train_Y
        # test_Y_ = np.ones(len(test_Y)) - test_Y

        batch_idxs_train = FLAGS.train_length // FLAGS.ebatch
        batch_idxs_vali =   FLAGS.Vali_length // FLAGS.ebatch
        batch_idxs_test = FLAGS.test_length // FLAGS.ebatch

        max_acc_avg = 0
        max_vali_avg = 0

        #min = float(FLAGS.learning_rate_min)
        #max = float(FLAGS.learning_rate_max)

        counter = 0
        # graphs = tf.get_default_graph()
        graph = self.net.graph
        # graph.as_default()

        with tf.Session(graph = graph) as sess:

            self.net.saver.restore(sess, self.chkpt_file)
            print('Parameters were initialized')
            self.net.print_model()

            for epoch in range(1,2):
                learning_rate = FLAGS.lrate
                #start_time = time.time()  # 先记录这一步的时间
                counter_ = counter
                # 训练
                train_acc_avg = 0
                loss = 0
                for idx in range(0, batch_idxs_train):
                    train_X, train_Y = dota.train.next_batch(FLAGS.ebatch)
                    feed_dict = self.feed_data(train_X, train_Y, idx, 1.0, learning_rate)
                    loss, summary, train_acc = sess.run(
                            [self.net.loss, self.net.summary_op, self.net.accuracy],
                            feed_dict=feed_dict)
                    self.train_writer.add_summary(summary, counter)
                    counter += 1
                    train_acc_avg += train_acc
                print('Epoch %d : Train Accuracy = %.2f ;lr=%.8f;loss=%.5f' %
                              (epoch,100* train_acc_avg/batch_idxs_train,learning_rate,loss))

                if epoch % 1 == 0:
                    #  验证
                    vali_acc_avg = 0
                    self.net.is_training = False
                    for idx in range(0, batch_idxs_vali):
                        vali_X, vali_Y = dota.validation.next_batch(FLAGS.ebatch)
                        feed_dict = self.feed_data(vali_X, vali_Y, idx, 1.0,learning_rate)
                        loss,summary,test_acc_ = sess.run(
                            [self.net.loss, self.net.summary_op, self.net.accuracy],
                            feed_dict=feed_dict, options=run_options)
                        self.test_writer.add_summary(summary, counter_)
                        counter_ += batch_idxs_train//batch_idxs_test
                        #平均和最优
                        vali_acc_avg+=test_acc_
                    vali_acc_avg /= batch_idxs_vali
                    if vali_acc_avg > max_vali_avg:
                        max_vali_avg = vali_acc_avg
                    #epoch_duration = time.time() - start_time
                    print('Epoch %d done:    Vali Acccuracy (avg) = %.2f , Top (avg) = %.5f '%
                          (epoch, 100 * vali_acc_avg, 100*max_vali_avg))
                    self.net.is_training = True
            test_acc_avg = 0
            counter_  = 0
            for idx in range(0, batch_idxs_test):
                test_X, test_Y = dota.test.next_batch(FLAGS.ebatch)
                feed_dict = self.feed_data(test_X, test_Y, idx, 1.0, FLAGS.lrate)
                tst,loss, summary, test_acc_ = sess.run(
                                [self.net.model.x_fc,self.net.loss, self.net.summary_op, self.net.accuracy],
                                feed_dict=feed_dict, options=run_options)
                # self.test_writer.add_summary(summary, counter_)
                counter_ += batch_idxs_train // batch_idxs_test
                # 平均和最优
                test_acc_avg += test_acc_
            test_acc_avg /= batch_idxs_test
            print('Test done,',test_acc_avg,loss)




    def feed_data(self,data,label,idx,keep_prob,learning_rate):
        if keep_prob != 1.0:
            is_training=True,
        else:
            is_training=False,
        return  {
            self.net.x: data,
            self.net.y: label,
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



