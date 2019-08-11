import gzip
import pickle
import numpy as np
from sklearn import svm

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

def read_all(file_name):
    f = gzip.open(file_name, 'rb')
    data_set, label_set = pickle.load(f, encoding='latin1')
    f.close()
    # label_set = dense_to_one_hot(label_set, num_classes=2)
    toZip = list(zip(data_set, label_set))
    #random.shuffle(toZip)
    data_set, label_set = map(list, zip(*toZip))
    return  data_set, label_set

def split(data_set):
    label = []
    data = []
    for datas in data_set:
        for i in range(20):
            data.append(datas[i*51:i*51+50])
            if datas[i*51+50] == 1:
                label.append(1)
            else:
                label.append(0)
    return data,label

def features_test():
    datas,labels = read_all('data_set/DotaTMzScore.set')
    data,label = split(datas)
    n = len(data)//4*3
    data_test = data[n:]
    label_test = label[n:]
    data_train = data[0:n]
    label_train = label[0:n]
    from sklearn import preprocessing
    # 范围0-1缩放标准化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_scaler = min_max_scaler.fit_transform(data)
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression, Ridge

    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(data_train, label_train)
    for i in range(50):
        if rfe.ranking_[i] > 30:
            rfe.support_[i] = False
            # c += 1
        else:
            rfe.support_[i] = True
            print(i)


            # n_epoch = 100
    # label_predict = [0]*len(data_test)
    # for i in range(n_epoch):
    #     base_clf = RandomForestClassifier(n_estimators=48, max_depth=7, min_samples_split=20)
    #     clf = BaggingClassifier(base_clf, n_estimators=40)
    #     clf.fit(data_train, label_train)
    #     label_predict += clf.predict(data_test)
    #     label_pro = clf.predict_proba(data_test)

    # param_test1 = {'max_depth':[n for n in range(1, 20, 2)],
    #                'min_samples_split':[n for n in range(2,41,2)]}
    # # param_test1 = {'n_estimators':[n for n in range(30, 60, 2)]}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators = 48,oob_score=True,
    #                                                          min_samples_leaf=20,
    #                                                          max_features='sqrt', random_state=10,
    #                                                          ),
    #                     param_grid=param_test1,scoring = 'roc_auc', cv=5)
    # gsearch1.fit(data_train, label_train)
    # for its in gsearch1.grid_scores_:
    #     print(its)
    # print(gsearch1.best_params_, gsearch1.best_score_)

    # label_predict = [0]
    # num = 0
    # correct_num = 0
    # while num < len(label_test):
    #     if label_predict[num] < n_epoch//2:
    #         label_predict[num] = 0
    #     else:
    #         label_predict[num] = 1
    #     if label_test[num] == label_predict[num]:
    #         correct_num += 1
    #     num += 1
    # print("预测准确率为：" + str(correct_num) + "/" + str(len(label_test))
    #       + "=" + str(correct_num / len(label_test)))

    # model = SelectFromModel(clf, prefit=True)
    # X_new = model.transform(data)
    # mask_features = model.get_support()
    # scores = [0]*51
    # sums = [0]*40
    # for i in range(40):
    #     print(list(mask_features[i*51:i*51+51]))
    #     sums[i]= np.sum(np.asarray(mask_features[i*51:i*51+51]))
    #     for j in range(51):
    #         if mask_features[i*51+j] == True:
    #             scores[j]+=1
    # print(list(mask_features[39*51+51:]))
    # print(scores)
    # print(sums[0:20])
    # print(sums[20:])
features_test()