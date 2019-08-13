import gzip
import pickle
import numpy as np
import random



teamCount = [0]*20

def dumpItems(dataitem,labelitem):
    print('Before:',len(dataitem))
    error = [-1]
    i = 0
    for items in dataitem:
        flag = 0
        count = 0
        teams = [-1] * 2
        for n in range(0,20):
            if items[n*1100]!=0 or items[n*1100+400]!=0 or items[n*1100+800]!=0 :
                teamCount[n] += 1
                if n == int(labelitem[i]):
                    flag = 1
                teams[count] = n
                count += 1
        if flag == 0:
            print('Error:',i,' 标签：',labelitem[i],'实际：',teams,'出现：',count)
            error.append(i)
        i += 1
    p = 0
    for err in error:
        if err!= -1:
            dataitem = np.delete(dataitem,  err-p+1, 0)
            labelitem = np.delete(labelitem, err-p+1, 0)
        p += 1

    toZip = list(zip(dataitem, labelitem))
    random.shuffle(toZip)
    datass, labelss = map(list, zip(*toZip))
    dataitem = np.asarray(datass, dtype=float)
    labelitem = np.asarray(labelss, dtype=float)
    print('After:',len(dataitem),len(labelitem))

    return dataitem,labelitem

def dumpDouble(dataitem,labelitem):
    print('Before:',len(dataitem))
    error = [-1]
    count = 0
    for i in range(0,len(dataitem)):
        print(i/len(dataitem))
        for j in range(i+1,len(dataitem)):
            flag = 0
            zeroflag = 0
            for n in range(0,len(dataitem[0])):
                if dataitem[i][n] == dataitem[j][n]:
                    flag += 1
                if dataitem[i][n] != 0:
                    zeroflag = 1
            if flag >= len(dataitem[0])-2 or zeroflag == 0:
                if zeroflag == 0:
                    print(i,'all zero!')
                    if i not in error:
                        error.append(i)
                    else:
                        print(i, 'has recorded')
                    count += 1
                else:
                    print(i,j,'in common')
                    print(list(dataitem[i]))
                    print(list(dataitem[j]))
                    if j not in error:
                        error.append(j)
                    else:
                        print(j,'has recorded')
                    count+=1
    p = 0
    for err in error:
        if err!= -1:
            dataitem = np.delete(dataitem,  err-p+1, 0)
            labelitem = np.delete(labelitem, err-p+1, 0)
        p += 1
    print('After:',len(dataitem))
    print(len(dataitem))
    toZip = list(zip(dataitem, labelitem))
    random.shuffle(toZip)
    datass, labelss = map(list, zip(*toZip))
    dataitem = np.asarray(datass, dtype=float)
    labelitem = np.asarray(labelss, dtype=float)
    return dataitem,labelitem

def double(dataitem,labelitem):
    count = [0]*20
    dataNew =[]
    labelNew = []
    j = 0
    for i in range(0,len(dataitem)):
        for n in range(0,20):
            if dataitem[i][n*1100]!=0 or dataitem[i][n*1100+400]!=0 or dataitem[i][n*1100+800]!=0 :
                count[n] += 1

                if n == 19 and j<10:
                    dataNew.append(list(dataitem[i]))
                    labelNew.append(list(labelitem)[i])
                    j+=1
    print(count)

    dataNew+=list(dataitem)
    labelNew+=list(labelitem)

    toZip = list(zip(dataNew[:-100], labelNew[:-100]))
    random.shuffle(toZip)
    datass, labelss = map(list, zip(*toZip))
    datass+=dataNew[-100:]
    labelss+=labelNew[-100:]
    dataNew = np.asarray(datass, dtype=float)
    labelNew = np.asarray(labelss, dtype=float)
    return dataNew,labelNew

def data_MinMax(Data):
    # 等级 KDA 参战率 伤害% 伤害 正/反补 经验/分钟 金钱/分钟 建筑伤害 英雄治疗
    mAx = [15,5,60,15,8000,40,300,300,700,6000,0]
    mIn = [15,5,60,15,8000,40,300,300,700,6000,0]

    n = 0
    print("Finding min&max...")
    for item in Data:
        #print(1.0 * n / len(Data) * 100.0, '%')
        for i in range(0, 20):
            items = Data[n]  # x需要测试长度 OK
            if items[i * 1100] == 0 or items[6 + i * 1100] == 0:
                continue
            #print(items[i * 1100:i*1100+1100])
            for j in range(0, 100):
                if items[1100 * i + j * 11] > 5:  # 等级阈值 需要改
                    for ij in range(0, 11):
                        if items[1100 * i+ j * 11 + ij] >= -1:
                            if items[1100 * i + j * 11 + ij] > mAx[ij]:
                                mAx[ij] = items[1100 * i + j * 11 + ij]
                            if items[1100 * i + j * 11 + ij] < mIn[ij]:
                                mIn[ij] = items[1100 * i + j * 11 + ij]
                else:
                    items[1100 * i  + j * 11:1100 * i + j * 11 + 11] = [0] * 11
        n += 1
    print(mAx)
    print(mIn)
    f = open('mInmAx.txt', 'a+')
    for u in mAx:
        f.write(str(u) + ' ')
    f.write('\n')
    for u in mIn:
        f.write(str(u) + ' ')
    f.write('\n')  # MinMax
    f.close()

    print(list(Data[0]))
    print('Processing...')
    Datanew = []
    for items in Data:
        infos = [0.0] * 22000
        for i in range(0, 20):
            if items[i * 1100] == 0 and items[i * 1100 + 1] == 0 and items[i * 1100 + 700] == 0 and items[i * 1100 + 550]==0:
                continue
            for j in range(0, 100):
                if items[1100 * i + j * 11] != 0:
                    for ij in range(0, 11):  # 不能是0
                        infos[i * 1100 + j * 11 + ij] = (items[i * 1100 + 11 * j + ij] - mIn[ij]) / (mAx[ij] - mIn[ij])
        Datanew.append(infos)
    # f = open('TesT.txt', 'a+')
    # for u in Datanew[0]:
    #    f.write(str(u) + ' ')
    # f.write('\n')
    # f.close()
    print(list(Datanew[0]))

    return Datanew

def compress2200(dataitem,labelitem):
    dataNew = []
    labelNew = []

    for i in range(0,len(dataitem)):
        flag = -1
        infos = [0] * 2200
        label = -1
        for n in range(0,20):
            if dataitem[i][n*1100]!=0 or dataitem[i][n*1100+1]!=0 or dataitem[i][n*1100+500]!=0 or dataitem[i][n*1100+700]!=0:
                flag+=1
                infos[flag*1100:flag*1100+1100] = dataitem[i][n*1100:n*1100+1100]
                if n==labelitem[i] and flag == 0:
                    label = 0
                if n==labelitem[i] and flag == 1:
                    label = 1
                print(len(infos))
        if flag == -1:
            print('Error')
        dataNew.append(list(infos))
        labelNew.append(label)
    toZip = list(zip(dataNew, labelNew))
    random.shuffle(toZip)
    datass, labelss = map(list, zip(*toZip))
    dataNew = np.asarray(datass, dtype=float)
    labelNew = np.asarray(labelss, dtype=float)
    return dataNew,labelNew

def compress2202(dataitem,labelitem):
    dataNew = []
    labelNew = []

    for i in range(0,len(dataitem)):
        flag = -1
        infos = [0] * 2202
        label = -1
        for n in range(0,20):
            if dataitem[i][n*1100]!=0 or dataitem[i][n*1100+1] or dataitem[i][n*1100+500]!=0 or dataitem[i][n*1100+700]!=0:
                flag+=1
                infos[flag*1100+flag+1:flag*1100+1100+flag+1] = dataitem[i][n*1100:n*1100+1100]
                infos[flag*1100+flag] = (21-n)/21
                if n==labelitem[i] and flag == 0:
                    label = 0
                if n==labelitem[i] and flag == 1:
                    label = 1
        if flag == -1:
          print('Error')
        dataNew.append(list(infos))
        labelNew.append(label)
    toZip = list(zip(dataNew, labelNew))
    random.shuffle(toZip)
    datass, labelss = map(list, zip(*toZip))
    dataNew = np.asarray(datass, dtype=float)
    labelNew = np.asarray(labelss, dtype=float)
    return dataNew,labelNew



def main():
    filename = '2data22000smallraw.pkl.gz'  #2data22000smallraw.pkl.gz,dota0505M22000.pkl.gz,dota0505raw.pkl.gz
    f = gzip.open(filename, 'rb')
    # dotaGenerate1dataTXsmallFull.pkl.gz
    # f = gzip.open('dota0505M22000.pkl.gz', 'rb')
    # f = gzip.open('Dota/dota0505VE22000.pkl.gz', 'rb')
    dataitem, labelitem = pickle.load(f, encoding='latin1')
    #print(list(dataitem[19]))
    #print(list(dataitem[91]))

    dataitem = data_MinMax(dataitem)
    #dataitem,labelitem = dumpItems(dataitem,labelitem)
    #dataitem,labelitem = compress2202(dataitem,labelitem)
    #dataitem, labelitem = dumpDouble(dataitem, labelitem)
    #dataitem, labelitem = double(dataitem, labelitem)


    print('After:',len(dataitem),len(labelitem))
    while(len(dataitem)%5!=0):
        dataitem = np.delete(dataitem, 1, 0)
        labelitem = np.delete(labelitem, 1, 0)
        print('Deleting...1,   len:',len(dataitem))


    All = dataitem, labelitem
    p = pickle.dumps(All, 2)  # 生成pkl.gz文件就和theano中的一样
    #s = gzip.open('dota0505M22000.pkl.gz', 'wb')  # save as .gz,dota0505M2202.pkl.gz
    filenameNew = '2dota2202.pkl.gz'
    s = gzip.open(filenameNew, 'wb')
    s.write(p)
    s.close()

main()
'''   
 i = 0
    for matches in dataitem:
        for j in range(i+1,len(dataitem)):
            if dataitem[i].all ==dataitem[j].all:
                print('23121241432525')
        i+=1

            f = open('TesT.txt', 'a+')
            for u in items:
                f.write(str(u) + ' ')
            f.write('\n')
'''
