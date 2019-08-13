'''

快乐爬虫
爬取dotamax上目前的队伍排名 然后按排名获取各个队伍的最近比赛
维护一个比赛id的访问列表 以防断线重启

作者：戚朕
时间：2019年8月13日21:32:13
'''

import urllib.request
import re
import socket
from bs4 import BeautifulSoup
import http.cookiejar


Host = "http://www.dotamax.com"
topList = 'TeamTop0813.txt'
visitList = 'visit0813.txt'
data_set = 'data_set0813_20.txt'
label_set = 'label_set0813_20.txt'
teamGame = 'teamGame0813/'

def writeData(file,data):
    f = open(file, 'a+')
    if type(data)==list:
        for adata in data:
            f.write(str(adata)+' ')
    else:
        f.write(str(data))
    f.write('\n')
    f.close()

def readList(file):
    data = []
    try:
        f = open(file)
    except:
        f = open(file, 'a+')
        f.close()
    with open(file) as f:
        for line in f:
            if line!='\n':
             data.append(line.split())
    return data

def getPage(url):
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Encoding':'utf-8',
               'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'Cookie':' _ga=GA1.2.337723387.1565699744; _gid=GA1.2.890941211.1565699744; pkey=MTU2NTY5OTc2OC4wN3FpemhlbjgxNl8yYmlnaWF4dnVia21oc2lkYQ____; cookie="gAJ9cQFVD2RqYW5nb190aW1lem9uZVUNQXNpYS9TaGFuZ2hhaXMu:1hxW67:5-0xkbNkd0j7cpSBOSIKMTNIXnc"',
               'Host': 'www.dotamax.com',
               #'Upgrade-Insecure-Requests': '1',
               'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
               'Referer': url
               # 'Chrome/51.0.2704.63 Safari/537.36',
               }
    retryTimes = 200

    while retryTimes>0:
        try:
            timeout = 60
            socket.setdefaulttimeout(timeout)
            req = urllib.request.Request(url=url, headers=headers)
            cj = http.cookiejar.CookieJar()
            #res = urllib.request.urlopen(req)
            #html = res.read()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            res = opener.open(req)
            html = res.read()
            html = html.decode('UTF-8')
            return html
        except:
            print('.')
            retryTimes-=1

def getMatchData(url):
    html = getPage(url)
    soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
    ids = re.findall('href="/player/detail/.*?</tr>', html, re.S)[1:]
    pwin = soup.find('span', class_=re.compile("hero-title")).find('font').get_text()
    teamIDs = re.findall('href="/match/tour_team_detail/.*?"', html, re.S)
    if len(teamIDs)<2:
        teamID1=0
        teamID2=0
    else:
        teamID1 = teamIDs[0].split('team_id=')[1].split('"')[0]
        teamID2 = teamIDs[1].split('team_id=')[1].split('"')[0]
    if pwin == "夜魇获胜":
        try:
            winlose = [-1*int(teamID1),1*int(teamID2)]
        except:
            try:
                winlose = [-1 * float(teamID1), 1 * float(teamID2)]
            except:
                print('>>>Error:数值问题')
                return -1
    else:
        try:
            winlose = [1*int(teamID1),-1*int(teamID2)]
        except:
            try:
                winlose = [1 * float(teamID1), -1 * float(teamID2)]
            except:
                print('>>>Error:数值问题')
                return -1

    state = [0] * 100
    id = []
    for i in range(0,10):  # 在页面中寻找id为playerid的 记录表现
        try:
            rawID = ids[i].split('detail/')[1].split('" >')[0]
        except:
            print('>>>Error:页面解析有问题')
            return -1
        if rawID =='132538905': #修改某个选手
            rawID = '86745912'
        id.append(rawID)
        # print(id)
        if ids[i].find('herolevel">') != -1:
            state[0+10*i] = ids[i].split('herolevel">')[1].split('<')[0]
            if int(state[0+10*i]) < 3:
                print('>>>Error:数值为0')
                return 0
        else:
            try:
                state[0+10*i] = ids[i].split('> ')[1].split('\n')[0]
                if int(state[0 + 10 * i]) < 3:
                    print('>>>Error:数值为0')
                    return 0
            except:
                print('>>>Error:页面解析有问题')
                return -1
        state[1+10*i] = ids[i].split('a868">')[1].split("<")[0]
        states = re.findall('number">\d*.*\d%?</td>', ids[i], re.S)[0].split('>')
        state[2+10*i] = states[1][:-5]  # 参战率
        state[3+10*i] = states[3][:-5]  # 伤害
        state[4+10*i] = states[5][:-4]  # 伤害
        Chufa = states[7][:-4].split('/')  # 正反补
            # if Chufa[1] == '0':
            #   Chufa[1] = '1'
        state[5+10*i] = int(Chufa[0]) + int(Chufa[1])
        state[6+10*i] = states[9][:-4]  # 经验
        state[7+10*i] = states[11][:-4]  # 金钱
        state[8+10*i] = states[13][:-4]  # 建筑
        state[9+10*i] = states[15][:-4]  # 治疗
            # for j in range(0,8):
            #     state[j+2]=states[i].split()
            # allData = [0]*16580

    print('>>>已找到选手ID数：',len(id))
    #print(state)
    print('>>>Done,',winlose)
    return id,state,winlose

#getMatchData('http://www.dotamax.com/match/detail/3295677794')

def getGameList(url,teamNum):
    html = getPage(url)
    teamRank = [0]*8
    teamRank[0] = -1
    teamRank[1] = re.findall('700;">\d+\.+\d*%</font>', html, re.S)[0].split('>')[1].split('%')[0]
    teamRank[2] = re.findall('3627">\d*\.*\d?</font>', html, re.S)[0].split('>')[1].split('<')[0]
    gameidOne = re.findall('/match/detail/.*?" style', html, re.S)[0].split('detail/')[1].split('\'')[0]
    newUrl = "http://www.dotamax.com/match/detail/"+gameidOne
    playerIds,_,winlose=getMatchData(newUrl)
    if winlose[0] == int(teamNum) or -1*winlose[0] == int(teamNum):
        teamRank[3:8] = playerIds[0:5]
    elif winlose[1] == int(teamNum) or -1*winlose[1] == int(teamNum):
        teamRank[3:8] = playerIds[5:10]
    p=1
    while p<11:
        pageUrl = url + "&p="+str(p)
        html = getPage(pageUrl)
        gameids = re.findall('/match/detail/.*?" style', html, re.S)
        for gameId in gameids:
            gameid = gameId.split('detail/')[1].split('\'')[0]
            print(gameid)
            f = open(teamGame+str(teamNum) + '.txt', 'a+')
            f.write(gameid + '\n')
            f.close
        p += 1
    return teamRank

def makeToplist():
    url="http://www.dotamax.com/match/tour_famous_team_list/?time=all"
    teamInfo = [0]*9
    html=getPage(url)
    teams = re.findall('team_id=.*?;',html,re.S)
    htmlNew = getPage("http://www.dotamax.com/match/tour_famous_team_list/?skill=&ladder=&time=all&p=2")
    team21 = re.findall('team_id=.*?;',htmlNew,re.S)[0]
    teams.append(team21)
    team22 = re.findall('team_id=.*?;',htmlNew,re.S)[1]
    teams.append(team22)

    print(teams)
    i = 0
    for items in teams:
        teamInfo[0] = teams[i].split('=')[1].split('\'')[0]
        newUrl = "http://www.dotamax.com/match/tour_team_detail/?team_id=" + str(teamInfo[0])
        teamInfo[1:9] = getGameList(newUrl,teamInfo[0])[:]
        teamInfo[1] = 21-i
        #9个数据分别为：队伍ID，名次补码，胜率，MMR，五名队员
        writeData(topList,teamInfo)
        i=i+1

def readFirstList():
    Teams = []
    try:
        f = open(topList)
    except:
        makeToplist()
    with open(topList) as f:
        for line in f:
            Teams.append(line.split())
    Visit = []
    f = open(visitList, 'a+')
    f.close()
    try:
        with open(visitList) as f:
            for line in f:
                Visit.append(line.split()[0])
    except:
        Visit.append('0')
    return Teams,Visit

def getRecentGame(Teams,gameID):
    print('获取数据：Game ID:',gameID)
    url = 'http://www.dotamax.com/match/detail/'+str(gameID)
    result = getMatchData(url)
    if result == -1 or result == 0:
        return  -1
    playerIds, gameData, winlose = result #知道了单场比赛双方 根据返回winlose遍历两个队伍 需要打开文件读比扫列表
    recentGameNum = 20
    if int(winlose[0])>0:
        whowin = 0
    else:
        whowin = 1
    team1ID = abs(int(winlose[0]))
    team2ID = abs(int(winlose[1]))
    flag = 0
    error = []
    teamData = []
    for team in Teams:
        if int(team[0]) == team1ID:
            flag+=1
            i=0
            teamData.append(team)
            for player in playerIds:#尝试一下不把选手和队伍锁定，每次出现不同时更新选手信息
               if str(player) in team:
                   i+=1
               #else:
                #   print('>>>增加选手信息:'+str(player)+str(team))
                 #  error.append('>>>Error:选手不在队伍里：'+str(player)+str(team))
            if i<3:
                flag -= 1
            if i!=5:
                print('>>>选手变动，本次：',playerIds,'列表中：',team)
        if int(team[0]) == team2ID:
            flag+=1
            i=0
            teamData.append(team)
            for player in playerIds:
               if str(player) in team:
                   i+=1
               #else:
                #   print('>>>增加选手信息:'+str(player)+str(team))
                 #  error.append('>>>Error:选手不在队伍里：'+str(player)+str(team))
            if i<3:
                flag -= 1
            if i != 5:
                print('>>>选手变动，本次：', playerIds, '列表中：', team)
    if flag != 2:
        if len(error)>1:
            print(error)
        else:
            print('>>>Error:大部分选手不在队伍列表中:',winlose)
        return -1

    team1list = readList(teamGame+str(team1ID)+'.txt')
    team2list = readList(teamGame+str(team2ID)+'.txt')
    teamIDs = [team1ID,team2ID]
    teamlists = [team1list,team2list]
    i = 0
    oneData = [0]*(recentGameNum*104)
    #102含义: A vs. B 的比赛：
    # A1*10 A2*10 A3*10 A4*10 A5*10 WinLose anotherTeam B1*10 B2*10 B3*10 B4*10 B5*10 WinLose anotherTeam 共104
    # 以上的数据*要收集的比赛场数

    for teamlist in teamlists:
        flag = 0
        count = 0
        errorCount = 0
        for games in teamlist:
            if flag == 1:
                print('>>>比赛方：',i,'，最近第',count,'场：',str(games[0]))
                try:
                    resu = getMatchData('http://www.dotamax.com/match/detail/'+str(games[0]))
                except:
                    continue
                if resu == 0 or resu == -1:
                    continue
                playerIdOne,gameDataOne,WinLose = resu
                if int(WinLose[0]) == int(teamIDs[i]) :
                    win = 1
                    j =0
                    anotherTeam = abs(int(WinLose[1]))
                elif -int(WinLose[0]) == int(teamIDs[i]):
                    win = -1
                    j = 0
                    anotherTeam = abs(int(WinLose[1]))
                elif int(WinLose[1]) == int(teamIDs[i]):
                    win = 1
                    j = 1
                    anotherTeam = abs(int(WinLose[0]))
                elif -int(WinLose[1]) == int(teamIDs[i]):
                    win = -1
                    j = 1
                    anotherTeam = abs(int(WinLose[0]))
                else:
                    continue
                playerflag = 2#1
                for player in playerIdOne[j*5:j*5+5]:
                    if player not in playerIds:
                        playerflag -=1#= 0
                if playerflag >= 0:
                    oneData[count*104+i*52:count*104+i*52+50] = gameDataOne[j*50:j*50+50]
                    oneData[count * 104 + i * 52+50] = win
                    oneData[count * 104 + i * 52+51] = anotherTeam
                    #data[i].append(gameDataOne[j*5:j*5+5]+win)
                    count+=1
                else:
                    print('>>>Error:','本场比赛出现选手变动')
                    errorCount+=1
                    if errorCount == 5:
                        break
            if count == recentGameNum:
                break
            if games[0] == str(gameID):
                flag = 1
        i+=1
        if count!=recentGameNum:
            print('数据不足')
        if count<10:
            print('Error:数据不足')
            return  -1

    print('*一条数据：')
    print(len(oneData))
    print(oneData)
    return oneData,whowin,teamData


def getStart():

    Teams,Visit = readFirstList()
    gameID = []

    for i in range(22):
        print('=============',i,'============')
        with open(teamGame+Teams[i][0]+'.txt') as f:
            for line in f:
                newID = line.replace('\n','')
                if newID not in Visit:
                    if int(newID)<2200000000:
                        writeData(visitList, newID)  # 需要判定条件
                        continue
                    resu = getRecentGame(Teams,newID)
                    if resu!=-1:
                            data,winlose,twoTeam = resu #数据和标签
                            data = data + twoTeam[0][0:3]+twoTeam[1][0:3]#应该是[1:4]没有改 后面预处理再改
                            writeData(data_set,data)
                            writeData(label_set,winlose)
                            writeData(visitList, newID+' mark')
                    else:
                        writeData(visitList,newID)#需要判定条件
                    Visit.append(newID)
                else:
                    print(newID,'已存在')



getStart()
