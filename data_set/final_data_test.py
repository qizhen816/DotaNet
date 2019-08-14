import gzip
import pickle
import numpy as np
import random


def check_(all):
    data, label = all
    print('Before:', len(data))
    print('Start Now')
    error = [-1]
    count = 0
    for i in range(0, len(data)):
        zerocount = 0
        for j in range(i + 1, len(data)):
            flag = 0
            zerocount = 0
            for n in range(0, len(data[0])):
                if data[i][n] == data[j][n] and n<600:
                    zerocount += 1
            if zerocount > 500:
                count +=1
                print(i,j,zerocount,count)
    print('After:', len(data))
    print(error)
    return all

file_name = '2data_0821.set'
f = gzip.open(file_name, 'rb')
all = pickle.load(f, encoding='latin1')
data_set, label_set = all
f.close()
check_([data_set,label_set])