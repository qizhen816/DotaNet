import gzip
import pickle
import numpy as np
import random

file_name_1 = '2data_0821.set'#'2data_0821.set'#'DotaTMzScore.set'
file_name_2 = 'DotaTMzScore180814.set'#'DotaTMzScore180814.set'#'DotaTMzScore1.set'

def read(file_name,a,b):
    f = gzip.open(file_name, 'rb')
    all = pickle.load(f, encoding='latin1')
    data_set, label_set = all
    f.close()
    toZip = list(zip(data_set, label_set))
    data_set, label_set = map(list, zip(*toZip))
    data_set = np.asarray(data_set,dtype='float32')
    label_set = np.asarray(label_set,dtype='float32')
    train = [data_set[0:a],label_set[0:a]]
    vali = [data_set[a:b], label_set[a:b]]
    test = [data_set[b:],label_set[b:]]
    return train,vali,test

def save(all,filename):
    data_set, label_set = all
    toZip = list(zip(data_set, label_set))
    data_, label_ = map(list, zip(*toZip))
    Data = np.asarray(data_, dtype=float)
    Lable = np.asarray(label_, dtype=float)
    All = Data, Lable
    p = pickle.dumps(All, 2)
    print('Storing...')
    s = gzip.open(filename, 'wb')  # save as .gz
    s.write(p)
    s.close()
    print(filename, 'complete!')

train_1,vail_1,test_1 = read(file_name_1,2580,2780) # 1340,1440
train_2,vail_2,test_2 = read(file_name_2,870,970) # 1240,1340

data = np.concatenate([train_1[0],train_2[0]])
data = np.concatenate([data,vail_1[0]])
data = np.concatenate([data,vail_2[0]])
data = np.concatenate([data,test_1[0]])
data = np.concatenate([data,test_2[0]])

# label = train_1[1].append(train_2[1]).append(vail_1[1]).append(test_1[1])
label = np.concatenate([train_1[1],train_2[1]])
label = np.concatenate([label,vail_1[1]])
label = np.concatenate([label,vail_2[1]])
label = np.concatenate([label,test_1[1]])
label = np.concatenate([label,test_2[1]])


def check_(all):
    data, label = all
    print('Before:', len(data))
    print('Start Now')
    for i in range(0, len(data)):
        flag = 0
        for j in range(10*51):
            if data[i][j] !=0:
                flag = 1
        if flag == 0:
            print(1)
    return all
check_([data,label])
save([data,label],'2data_190814.set')