from config import *
import numpy as np
import tensorflow as tf
import gzip
import pickle
import collections
from sklearn import preprocessing


FLAGS = tf.app.flags.FLAGS



def dense_to_one_hot(labels_dense, num_classes=2):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    if num_classes != 1:
        for i in range(0, num_labels):
            labels_one_hot[i][int(labels_dense[i])] = 1
    else:
        for i in range(0, num_labels):
            labels_one_hot[i] = int(labels_dense[i])
    return labels_one_hot

def read_all(file_name,train_num,validation_num,test_num,All = None):
    f = gzip.open(file_name, 'rb')
    all = pickle.load(f, encoding='latin1')
    if len(all) == 2:
        data_set, label_set = all
        f.close()
        label_set = dense_to_one_hot(label_set, num_classes=FLAGS.eout)
        toZip = list(zip(data_set, label_set))
        #random.shuffle(toZip)
        data_set, label_set = map(list, zip(*toZip))
        if test_num == -1:
            test_num = len(data_set)-train_num-validation_num
        data_set = np.asarray(data_set,dtype='float32')
        label_set = np.asarray(label_set,dtype='float32')

        train_data = data_set[:train_num]
        train_label = label_set[:train_num]
        # train_data = preprocessing.scale(train_data)
        scaler = preprocessing.StandardScaler().fit(train_data[:,0:2040])
        train_data_ = scaler.transform(train_data[:,0:2040])
        train_data_ = np.concatenate([(train_data_[:,0:2040]),train_data[:,2040:]],1)

        validation_data = (data_set[train_num:train_num+validation_num,0:2040])
        validation_data = scaler.transform(validation_data)
        validation_data = np.concatenate([validation_data,data_set[train_num:train_num+validation_num,2040:]],1)

        test_data = (data_set[0-test_num:,0:2040])
        test_label = label_set[0-test_num:]
        test_data = scaler.transform(test_data)
        test_data = np.concatenate([test_data,data_set[0-test_num:,2040:]],1)
        if All!= None:
            d,l = all
            l =  dense_to_one_hot(l, num_classes=FLAGS.eout)
            toZip = list(zip(d, l))
            test_datas, test_label = map(list, zip(*toZip))
            test_datas = np.asarray(test_datas, dtype='float32')
            test_label = np.asarray(test_label, dtype='float32')
            test_data = test_datas[:,0:2040]
            test_data = scaler.transform(test_data)
            test_data = np.concatenate([test_data, test_datas[:,2040:]], 1)


        train = dota_data(train_data_,train_label)
        validation = dota_data(validation_data,label_set[train_num:train_num+validation_num])
        test = dota_data(test_data,test_label)

        return collections.namedtuple('Datasets', ['train', 'validation', 'test'])(train,validation,test)


class dota_data:
    def __init__(self,data_dir,label_set):
        self._data = data_dir
        self._label = label_set
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(data_dir)

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def num_examples(self):
        return self._num_examples


    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size,shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._label = self.label[perm0]
         # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._label[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._label = self.label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            images_new_part = self._data[start:end]
            labels_new_part = self._label[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end],self._label[start:end]
