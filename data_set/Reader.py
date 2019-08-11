from config import *
import numpy as np
import tensorflow as tf
import gzip
import pickle
import collections

FLAGS = tf.app.flags.FLAGS



def dense_to_one_hot(labels_dense, num_classes=20):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    if num_classes != 1:
        for i in range(0, num_labels):
            labels_one_hot[i][int(labels_dense[i])] = 1
    else:
        for i in range(0, num_labels):
            labels_one_hot[i] = int(labels_dense[i])
    return labels_one_hot

def read_all(file_name,train_num,validation_num,test_num):
    f = gzip.open(file_name, 'rb')
    data_set, label_set = pickle.load(f, encoding='latin1')
    f.close()
    label_set = dense_to_one_hot(label_set, num_classes=FLAGS.eout)
    toZip = list(zip(data_set, label_set))
    #random.shuffle(toZip)
    data_set, label_set = map(list, zip(*toZip))
    if test_num == -1:
        test_num = len(data_set)-train_num-validation_num
    data_set = np.asarray(data_set,dtype='float32')
    label_set = np.asarray(label_set,dtype='float32')
    train = dota_data(data_set[:train_num],label_set[:train_num])
    validation = dota_data(data_set[train_num:train_num+validation_num],label_set[train_num:train_num+validation_num])
    test = dota_data(data_set[0-test_num:],label_set[0-test_num:])
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

# a = read_all(list_name=list_name,train_num=100,validation_num=100,test_num=100)
# a,b = a.train.next_batch(2)
# print(a[0],b[0])
# pil_im = Image.fromarray(a[0])
# pil_im.show()
# box = (b[0][0] - b[0][2],b[0][1] - b[0][3],b[0][0] + b[0][2],b[0][1] + b[0][3])
# region = pil_im.crop(box)
# region.show()