'''

数据预处理

作者：戚朕
时间：2019年8月13日21:32:13
'''

import numpy as np
import random
import gzip
import pickle
from sklearn import preprocessing

DataFile = ["data_set0724_20.txt"] #0724_20
LabelFile = ["labe_set0724_20.txt"] #0724_20
topList = 'TeamTop0724.txt' #0724

DataFile_new = ["data_set0828_20.txt"] #0724_20
LabelFile_new  = ["label_set0828_20.txt"] #0724_20
topList_new  = 'TeamTop0828.txt' #0724


class RawData(object):
    def __init__(self, data,label,topList):
        self.data = data
        self.label = label
        self.topList = topList

    def data_read(self):
        print('Reading...')
        label_read = []
        label_back = []
        data_read = []
        data_back = []
        j = 0
        for items in self.data:
            with open(items) as f:  # 读取txt里面的data
                for line in f:
                    if line == '\n':
                        break
                    data = line.split()
                    if j % 4 != 0:
                        data_read.append(data)
                    else:
                        data_back.append(data)
                    j += 1
                    # print(datallback)
        data_read = data_read + data_back
        print((data_read[-1]))
        i = 0
        for items in self.label:
            with open(items) as f:  # 读取txt
                for line in f:
                    if i % 4 != 0:
                        label_read.append(str(line[0:-1]))  # 一共20类
                    else:
                        label_back.append(str(line[0:-1]))
                    i += 1
                    if i == j:
                        break

        label_read = label_read + label_back

        labels = np.asarray(label_read, dtype=float)
        datas = np.asarray(data_read, dtype=float)
        toZip = list(zip(datas, labels))
        random.shuffle(toZip)
        random.shuffle(toZip)
        datass, labelss = map(list, zip(*toZip))
        Data = np.asarray(datass, dtype=float)
        Lable = np.asarray(labelss, dtype=float)
        print('Ok,data length:',len(Data),'label length:',len(Lable))
        All = Data, Lable
        return All

    def change_order(self,all):
        data_set, label_set = all
        data_tmp = np.zeros((len(data_set),len(data_set[0])),dtype=float)
        #print(list(data_set[0]))
        #print(list(data_set[0][51:]))
        for i in range(0,len(data_set)):
            for j in range(0,20):
                p = 2*j
                data_tmp[i][j*51:j*51+51] = data_set[i][p*51:p*51+51]
            for j in range(20, 40):
                p = (j-20)*2+1
                data_tmp[i][j * 51:j * 51 + 51] = data_set[i][p * 51:p * 51 + 51]
            data_tmp[i][2040:] = data_set[i][2040:]
        #print(list(data_tmp[0]))
        #print(list(data_tmp[0][1020:]))
        all = data_tmp,label_set
        return all

    def change_teamdata(self,all):

        #print('横向规整,调整队伍数据的位置：')
        teams = []
        with open(self.topList) as f:
            for line in f:
                teams.append(line.split())
        data_set, label_set = all
        data_tmp = data_set[:]
        #data_tmp = np.zeros((len(data_set),len(data_set[0])),dtype=float)
        #data_tmp[:,0:3] = data_set[:,2040:2043]
        #data_tmp[:,3:2046] = data_set[:,0:2043]
        #data_tmp[:, 1023:1026] = data_set[:, 2043:2046]
        #data_tmp[:, 1026:2046] = data_set[:, 1020:2040]
        print('完成：')
        print('修改队伍数据：')
        print('Test(Before):', data_tmp[0][-3:], data_tmp[0][-6:-3])
        for i in range(0,len(data_tmp)):
            flag = 0
            for items in teams:
                if abs(float(items[0]) - data_tmp[i][-3])<0.1:
                    data_tmp[i][-3:] = items[1:4]
                    flag = 1
                if abs(float(items[0]) - data_tmp[i][-6])<0.1:
                    data_tmp[i][-6:-3] = items[1:4]
                    flag = 1
            if flag == 0:
                print('Error:',data_tmp[i][-3:],data_tmp[i][-6:-3])
        print('Test(After):', data_tmp[0][-3:], data_tmp[0][-6:-3])
        print('完成')
        all = data_tmp,label_set
        return all

    def check_five(self,all):
        print('规整数目：')
        data_set, label_set = all
        label_set = np.asarray(label_set, dtype=float)
        data_set = np.asarray(data_set, dtype=float)
        while (len(label_set) % 5 != 0):
            data_set = np.delete(data_set, 1, 0)
            label_set = np.delete(label_set, 1, 0)
            print('Deleting...1,   len:', len(label_set))
        all = data_set,label_set
        print('完成')
        return all

    def storeData(self,filename,all,israndom = True):
        data_set,label_set = self.check_five(all)
        toZip = list(zip(data_set, label_set))
        if israndom == True:
            random.shuffle(toZip)
        data_, label_ = map(list, zip(*toZip))
        Data = np.asarray(data_, dtype=float)
        Lable = np.asarray(label_, dtype=float)
        All = Data, Lable
        p = pickle.dumps(All, 2)  # 生成pkl.gz文件就和theano中的一样
        print('Storing...')
        s = gzip.open(filename, 'wb')  # save as .gz
        s.write(p)
        s.close()
        print(filename,'complete!')

    def check_double(self,all):
        data,label = all
        print('Before:', len(data))
        print('Start Now')
        error = [-1]
        count = 0
        for i in range(0, len(data)):
            print(' Scanning:',i / len(data),'%')
            for j in range(i + 1, len(data)):
                flag = 0
                zeroflag = 0
                for n in range(0, len(data[0])):
                    if n - flag >=10:
                        break #0727 修改 加速
                    if data[i][n] == data[j][n]:
                        flag += 1
                    if data[i][n] != 0:
                        zeroflag = 1
                if flag >= len(data[0]) - 10 - 1000 or zeroflag == 0:
                    if zeroflag == 0:
                        print(i, 'all zero!')
                        if i not in error:
                            error.append(i)
                        else:
                            print(i, 'has recorded')
                        count += 1
                    else:
                        print(i, j, 'in common')
                        print(list(data[i]))
                        print(list(data[j]))
                        if j not in error:
                            error.append(j)
                        else:
                            print(j, 'has recorded')
                        count += 1
        p = 0
        for err in error:
            if err != -1:
                data = np.delete(data, err - p + 1, 0)
                label = np.delete(label, err - p + 1, 0)
            p += 1
        print('After:', len(data))
        toZip = list(zip(data, label))
        random.shuffle(toZip)
        datass, labelss = map(list, zip(*toZip))
        data_done = np.asarray(datass, dtype=float)
        label_done = np.asarray(labelss, dtype=float)
        all = data_done,label_done
        return all

    def check_zeros(self,all,maxZeros = 150):
        data, label = all
        error = [-1]
        j = 0
        print('去除零过多的数据：')
        for items in data:
            count1 = 0
            count2 = 0
            lens = len(items)
            #if lens != 2046 and lens != 2086:
                #print(list(items))
            for i in range(lens-8,0,-1):
                if float(items[i]) == 0.0:
                    count1 += 1
                else:
                    break
            for i in range((lens-8)//2-1,0,-1):
                if float(items[i]) == 0.0:
                    count2 += 1
                else:
                    break
            if count1>maxZeros or count2>maxZeros:
                print(j,count1,count2)
                error.append(j)
            j+=1
        p = 0
        for err in error:
            if err != -1:
                print('Deleting:',err - p + 1)
                data = np.delete(data, err - p + 1, 0)
                label = np.delete(label, err - p + 1, 0)
            p += 1
        print('After:',len(data))
        print('完成')
        all = data,label
        return all

    def zScore(self,all):
        data,label = self.check_five(all)

        n = 51
        # data_front = np.reshape(data[:,0:-6],[len(data)*40,n])
        # data_front_1 = data_front[:,:-1]
        # data_front_2 = data_front[:,-1]
        # data_front_2 = np.reshape(data_front_2,[-1,1])

        print('处理0：')
        imp = preprocessing.Imputer(missing_values=0,strategy='mean',verbose=0)
        imp.fit( data )
        data = imp.transform( data)
        print('Z-Score 标准化：')

        # data_front_1 = preprocessing.scale(data_front_1)
        # data_front = np.concatenate((data_front_1,data_front_2),axis=1)
        # data_front = np.reshape(data_front,[len(data),40*n])
        # data = np.concatenate((data_front,data[:,-6:]),axis=1)


        data = preprocessing.scale(data)
        print('均值:',list(data.mean(axis=0)))
        print('方差:',list(data.std(axis=0)))
        print('最大值:',list(data.max(axis=0)))
        print('最小值:',list(data.min(axis=0)))
        all = data,label
        return all,'DotaTM0707zScore.set'

    def MinMax(self,all):
        data,label = self.check_five(all)
        print('Min-Max 标准化：')
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        print('均值:',list(data.mean(axis=0)))
        print('方差:',list(data.std(axis=0)))
        print('最大值:',list(data.max(axis=0)))
        print('最小值:',list(data.min(axis=0)))
        all = data,label
        return all,'DotaTM0707minmax.set'

    def Normalization(self,all):
        data,label = self.check_five(all)
        # print('处理0：')
        # imp = preprocessing.Imputer(missing_values=0,strategy='mean',verbose=0)
        # imp.fit(data)
        # data = imp.transform(data)

        data_front = np.reshape(data[:,0:-6],[len(data)*40,41])
        data_front_1 = data_front[:,:-1]
        data_front_2 = data_front[:,-1]
        data_front_2 = np.reshape(data_front_2,[-1,1])
        data_front_1 = preprocessing.normalize(data_front_1,norm='l2',axis=0)
        data_front = np.concatenate((data_front_1,data_front_2),axis=1)
        data_front = np.reshape(data_front,[len(data),1640])
        data = np.concatenate((data_front,data[:,-6:]),axis=1)

        # data -= np.mean(data,axis = 0)

        print('l2正则化：')
        #data = preprocessing.normalize(data,norm='l2',axis=0)
        print('均值:',list(data.mean(axis=0)))
        print('方差:',list(data.std(axis=0)))
        print('最大值:',list(data.max(axis=0)))
        print('最小值:',list(data.min(axis=0)))
        all = data,label
        return all,'DotaTM0707norm.set'

    def test(self,all,lEn = 51):
        data,label = all
        testdata = data[50]
        '''
        for testdata in data:
            #print(list(testdata[0:3]))
            #print(list(testdata[3 + 3 * 51:3 + 3 * 51 + 51]))
            print(list(testdata[30 * 51:30 * 51 + 51]))
        '''
        n = (len(data[0])-6)//lEn
        for i in range(0,n):
            print(list(testdata[i*lEn:i*lEn+lEn]))
            if i == n//2-1:
                print('mid')
        print(list(testdata[-6:]))
        print('test done')

        #for i in range(20):
        #    print(list(testdata[1026+i*51:1026+i*51+51]))

    def make_double(self,all,type):
        data,label = self.check_five(all)
        trainPart = len(all[0])//4*3
        while trainPart%5!=0:
            trainPart+=1
        dataNew = []
        labelNew = []
        datalen = (len(data[0])-6)//2
        if type == 'dnn':
            for i in range(0,trainPart):
                dataTmp = [0] * len(data[0])
                dataTmp[0:datalen] = list(data)[i][datalen:datalen*2]
                dataTmp[datalen:datalen*2] = list(data)[i][0:datalen]
                dataTmp[datalen*2:datalen*2+3] = list(data)[i][datalen*2+3:datalen*2+6]
                dataTmp[datalen*2+3:datalen*2+6] = list(data)[i][datalen*2:datalen*2+3]
                labelNew.append(1-float(label[i]))
                dataNew.append(dataTmp)
        else:
            for i in range(0, trainPart):
                dataTmp = [0] * len(data[0])
                dataTmp[0:3] = list(data)[i][1023:1026]
                dataTmp[1023:1026] = list(data)[i][0:3]
                for j in range(0,10):
                    dataTmp[3+j*102:3+j*102+51] = list(data)[i][3+j*102+51:3+j*102+102]
                    dataTmp[3+j*102+51:3+j*102+102] = list(data)[i][3+j*102:3+j*102+51]
                labelNew.append(1 - float(label[i]))
                for j in range(0,10):
                    dataTmp[1026+j*102:1026+j*102+51] = list(data)[i][1026+j*102+51:1026+j*102+102]
                    dataTmp[1026+j*102+51:1026+j*102+102] = list(data)[i][1026+j*102:1026+j*102+51]
                labelNew.append(1 - float(label[i]))
                dataNew.append(dataTmp)
        #print(dataNew[0][0:10])
        #print(dataNew[0][1023:1033])
        #print(labelNew[0])
        dataNew+=list(data)[:trainPart]
        labelNew+=list(label)[:trainPart]
        toZip = list(zip(dataNew[:], labelNew[:]))
        random.shuffle(toZip)
        Data, Label = map(list, zip(*toZip))
        print('交换之后，训练集长度：',len(Data))
        Data += list(data)[trainPart:]
        Label += list(label)[trainPart:]
        Data = np.asarray(Data, dtype=float)
        Label = np.asarray(Label, dtype=float)
        print('总长度',len(Data))
        all = Data,Label
        return all

    def make_rnn(self,all,alpha = 5,isgamma = True,teamRank = False):
        data, label = all
        print('按照规则修改比赛数据：')
        teams = []
        with open(self.topList) as f:
            for line in f:
                teams.append(line.split())
        dataNew = np.zeros((len(data),2046),dtype=float)
        dataCount = 0
        for items in data:
            dataTmp = np.zeros((40*51+6),dtype=float)
            #id 范围： 0-21 与文件中下标一致 未在文件中出现的是22
            teamID1 = 21 - int(items[-5])
            teamID2 = 21 - int(items[-2])
            if teamRank == True:
                teamRank1 = (21 - float(teamID1))/21 + 1
                teamRank2 = (21 - float(teamID2))/21 + 1
            else:
                teamRank1 = 1
                teamRank2 = 1
            #寻找每场比赛对手的排名
            #首先是A 偶数下标
            for i in range(0,20):
                gameCount = i*2
                gameCount = 38 - gameCount #逆时序
                #测试 比赛数据为0但胜负不为0
                if items[52 * gameCount + 1] == 0 and items[52 * gameCount] == 0:
                    if int(items[gameCount*52+50]) !=0:
                        print(items[gameCount*52:gameCount*52+51])
                    continue
                # A vs. B 中 A独自场次中C的ID
                teamIDother = int(items[52*gameCount+51])
                p = 0
                for team in teams:
                    if int(team[0]) == teamIDother:
                        teamIDother = p
                        break
                    p += 1
                if p > 21:
                    teamIDother = 22
                dataTmp[gameCount*51+50] = items[gameCount*52+50]#胜负
                winlose = int(items[gameCount*52+50])
                gamma = winlose * (1 + winlose * (teamID2 - teamIDother) / 23) #C对于B的权重
                #dataTmp[gameCount * 51 + 50] = items[gameCount * 52 + 50] * gamma * winlose  # 胜负
                # 是否胜负乘以权重              50 51
                if isgamma == False:
                    gamma = 1
                #缩放
                if (gamma > 0 and gamma < 1) or (gamma < 0 and gamma > -1):
                    gamma /= alpha
                if gamma > 1 or gamma < -1:
                    gamma *= alpha
                decay = np.exp(-(38-gameCount)/15)
                dataTmp[gameCount*51:gameCount*51+50] = teamRank1 * items[gameCount*52:gameCount*52+50]*gamma*decay
                dataTmp[gameCount * 51 + 50] = gamma*decay
            for i in range(0,20):
                gameCount = i*2+1
                gameCount = 40 - gameCount #逆时序
                #测试 比赛数据为0但胜负不为0
                if items[52 * gameCount + 1] == 0 and items[52 * gameCount] == 0:
                    if int(items[gameCount*52+50]) !=0:
                        print(items[gameCount*52:gameCount*52+51])
                    continue
                # A vs. B 中 B独自场次中D的ID
                teamIDother = items[52*gameCount+51]
                p = 0
                for team in teams:
                    if int(team[0]) == teamIDother:
                        teamIDother = p
                        break
                    p += 1
                if p > 21:
                    teamIDother = 22 #?
                dataTmp[gameCount*51+50] = items[gameCount*52+50]#胜负
                winlose = int(items[gameCount*52+50])
                gamma = winlose * (1 + winlose * (teamID1 - teamIDother) / 23) #D对于A的权重
                #dataTmp[gameCount * 51 + 50] = items[gameCount * 52 + 50] * gamma * winlose # 胜负
                #是否胜负乘以权重
                if isgamma == False:
                    gamma = 1
                #缩放
                if (gamma > 0 and gamma < 1) or (gamma < 0 and gamma > -1):
                    gamma /= alpha
                if gamma > 1 or gamma < -1:
                    gamma *= alpha
                decay = np.exp(-(39-gameCount)/15)
                dataTmp[gameCount*51:gameCount*51+50] = teamRank2 * items[gameCount*52:gameCount*52+50]*gamma*decay
                dataTmp[gameCount * 51 + 50] = gamma*decay
            #复制后六位队伍信息 队伍排名都+1避免出现0
            dataTmp[-6:] = items[-6:]
            dataTmp[-2] += 1
            dataTmp[-5] += 1
            dataNew[dataCount][:] = dataTmp[:]
            dataCount+=1

        print('完成')
        All = dataNew,label
        return All

    def change_length(self,all,changeorder = True,length = 2040):
        data,label = all
        dataNew = np.zeros((len(data),length),dtype=float)
        dataNew[:,0:1020] = data[:,0:1020]
        dataNew[:,1020:2040] = data[:,1020:2040]
        if changeorder == True:
            dataNew_Tmp = dataNew[:]
            dataNew = np.zeros((len(data), length), dtype=float)
            for i in range(0,20):
                dataNew[:,i*51:i*51+51] = dataNew_Tmp[:,2*i*51:2*i*51+51]
            for i in range(0,20):
                dataNew[:,(i+20)*51:(i+20)*51+51] = dataNew_Tmp[:,(2*i+1)*51:(2*i+1)*51+51]
            if length == 2046 :
                dataNew[:,-6:] = data[:,-6:]
        print(list(dataNew[0]))
        print()
        return  dataNew,label

    def reverse(self,all,length = 2046):
        data,label = all
        dataNew = np.zeros((len(data),length),dtype=float)
        for i in range(0,20):
            dataNew[:,i*51:i*51+51] = data[:,(19-i)*51:(19-i)*51+51]
            dataNew[:,(i+20)*51:(i+20)*51+51] = data[:,(39-i)*51:(39-i)*51+51]
        dataNew[:, -6:] = data[:, -6:]
        return  dataNew,label

    def merge_reading(self,):
        visit0828 = []
        dup = []
        org = []
        with open('visit0828.txt') as f:
            for line in f:
                if len(line.split()) > 1 and line.split()[1] == 'mark':
                    visit0828.append(line.split()[0])
        count = 0
        with open('visit0724.txt') as f:
            for line in f:
                if len(line.split()) > 1 and line.split()[1] == 'mark':
                    if line.split()[0] in visit0828:
                        dup.append(count)
                        org.append(visit0828.index(line.split()[0]))
            count += 1
        print('Reading...')
        data_All = []
        label_All = []
        label_tmp = []
        data_ = []

        for items in LabelFile_new:
            with open(items) as f:  # 读取txt
                for line in f:
                    label_All.append(str(line[0:-1]))
        n = 0
        for items in DataFile_new:
            with open(items) as f:  # 读取txt
                for line in f:
                    data_All.append(line.split())
                    data_.append(line.split()[:-7])
                    if n == org[0]:
                        print(line.split())
                    n += 1
        for items in LabelFile:
            with open(items) as f:  # 读取txt
                for line in f:
                    label_tmp.append(str(line[0:-1]))
        i = 0
        count = 0
        for items in DataFile:
            with open(items) as f:  # 读取txt
                for line in f:
                    data_tmp = line.split()
                    data_tmp_ = data_tmp[:-7]
                    if data_tmp_ in data_:
                        print('SHIT!')
                        count += 1
                    else:
                        data_All.append(data_tmp)
                        label_All.append(label_tmp[i])
                    i += 1

        print("重复个数：",count)
        labels = np.asarray(label_All, dtype=float)
        datas = np.asarray(data_All, dtype=float)
        toZip = list(zip(datas, labels))
        random.shuffle(toZip)
        random.shuffle(toZip)
        datass, labelss = map(list, zip(*toZip))
        Data = np.asarray(datass, dtype=float)
        Lable = np.asarray(labelss, dtype=float)
        print('Ok,data length:',len(Data),'label length:',len(Lable))
        All = Data, Lable
        return All

    def check(self,all):
        data,label = all
        for dataOne in data:
            flag = 0
            for ii in range(0,20):
                #i = ii *2
                i = ii
                if dataOne[i*51] == 0.0:
                    flag = 1
                else:
                    if flag == 1:
                        print('eRRoR')
                        for t in range(0,20):
                            print(dataOne[t*51])
                        print('……………………………………')
            flag = 0
            for ii in range(0,20):
                #i = ii*2 + 1
                i = ii + 20
                if dataOne[i*51] == 0.0:
                    flag = 1
                else:
                    if flag == 1:
                        print('eRRoR')
                        for t in range(0,20):
                            print(dataOne[(t+20)*51])
                        print('……………………………………')

        print('Check Finish')

    def data_cut(self,all,match_count,leN = 51):
        data,label = all
        match_all = 20
        n = match_all - match_count
        mid = (len(data[0])-6)//2
        dataNew = np.zeros((len(data),leN*2*match_count+6),dtype=float)
        dataNew[:,0:match_count*leN] = data[:,n*leN:match_count*leN+n*leN]
        dataNew[:,match_count*leN:match_count*leN*2] = data[:,mid+n*leN:mid+match_count*leN+n*leN]
        dataNew[:,-6:] = data[:,-6:]
        print('切割之后，特征长度:',len(dataNew[0]))
        All = dataNew,label
        return All

    def delone(self,all):
        data,label = all



    def data_shot(self,all):
        data,label = all
        for i in range(0,40):
            j = i*41
            for n in range(0,5):
                data = np.delete(data, j + n * 8 + 4, 1)
                data = np.delete(data, j + n * 8 + 5, 1)
        all = data,label
        return all

    def pre_DNN(self):
        All = self.data_read()
        All = self.check_zeros(All)
        All = self.change_order(All) #变成 -A-B-3-3-
        All = self.change_teamdata(All) #变成 -3-?-3-?-
        self.test(All)
        return All

    def pre_CNN(self):
        All = self.data_read()
        All = self.check_zeros(All)
        All = self.change_teamdata(All) #变成 -3-?-3-?-
        #All = self.make_double(All)
        #print(list(All[1]))
        self.test(All)
        return All

    def pre_RNN(self):
        All = self.data_read()
        #All = self.merge_reading()
        All = self.check_zeros(All,200) #顺序正常
        #All = self.check_double(All)
        # alpha：缩放因子 isgamma:True每场比赛权值 teamRank:True全体乘排名的权值
        All = self.make_rnn(All,alpha = 3,isgamma = True,teamRank = False) #ABAB...AB33 #顺序正常
       # All = self.check_double(All)
        #self.check(All)
        All = self.change_teamdata(All) #顺序正常
        #All = self.make_double(All,'cnn')
        #print(list(All[0][1]))
        #self.test(All)
        return All


def merge_and_store(a,b,filename,israndom = True):
    data_a,label_a = a
    data_b,label_b = b
    data = list(data_a) + list(data_b)
    label = list(label_b) + list(label_b)
    toZip = list(zip(data, label))
    if israndom == True:
        random.shuffle(toZip)
    data_, label_ = map(list, zip(*toZip))
    #扩充
    trainPart = len(data_) // 4 * 3
    while trainPart % 5 != 0:
        trainPart += 1
    dataNew = []
    labelNew = []
    for i in range(0, trainPart):
            dataTmp = [0] * len(data_[0])
            dataTmp[0:1020] = list(data_)[i][1020:2040]
            dataTmp[1020:2040] = list(data_)[i][0:1020]
            dataTmp[2040:] = list(data_)[i][2040:]
            labelNew.append(1 - float(label_[i]))
            dataNew.append(dataTmp)
    dataNew += list(data_)[:trainPart]
    labelNew += list(label_)[:trainPart]
    toZip = list(zip(dataNew[:], labelNew[:]))
    random.shuffle(toZip)
    Data, Label = map(list, zip(*toZip))
    print('交换之后，训练集长度：', len(Data))
    Data += list(data_)[trainPart:]
    Label += list(label_)[trainPart:]

    Data = np.asarray(Data, dtype=float)
    Lable = np.asarray(Label, dtype=float)
    All = Data, Lable
    p = pickle.dumps(All, 2)  # 生成pkl.gz文件就和theano中的一样
    print('Storing...')
    s = gzip.open(filename, 'wb')  # save as .gz
    s.write(p)
    s.close()
    print(filename, 'complete!')


Obj = RawData(DataFile_new,LabelFile_new,topList_new)
All = Obj.pre_RNN()
All = Obj.change_length(All,changeorder=True,length=2046)
#All = Obj.data_cut(All,match_count=15,leN = 51)
#Obj.check(All)

#All = Obj.make_double(All,'dnn')

#Obj.storeData('DotaTM0724raw.set',All)
#All,filename = Obj.zScore(All)
#Obj.storeData(filename,All)
#All,filename = Obj.MinMax(All)
All = Obj.check_zeros(All,400)
#Obj.test(All)
All = Obj.reverse(All)
# All = Obj.data_shot(All)
Obj.test(All,lEn = 51)
#Obj.storeData(filename,All)

#All,_ = Obj.Normalization(All)
All,_ = Obj.zScore(All)
filename = 'DotaTM0707zScore.set'

All = Obj.make_double(All,'dnn')
#All = Obj.data_cut(All,match_count = 10,leN = 41)

#Obj.test(All)
#All = Obj.check_double(All)
Obj.storeData(filename,All)

# Obj_old = RawData(DataFile,LabelFile,topList)
# All_old = Obj_old.pre_RNN()
# All_old = Obj_old.change_length(All_old,changeorder=True,length=2046)
# All_old = Obj_old.reverse(All_old)
# All_old,filename_old = Obj_old.Normalization(All_old)
# merge_and_store(All,All_old,filename_old)