import urllib.request
from urllib import parse
import re
import socket
from bs4 import BeautifulSoup
import http.cookiejar
import requests
import rsa
import binascii
import os
from lxml import html
import base64

Host = "http://www.dotamax.com"
topList = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data_set/TeamTop190814.txt'
visitList = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data_set/visit190814.txt'
data_set = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data_set/data_set190814.txt'
label_set = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data_set/label_set190814.txt'
teamGame = 'E:/Vsprojs/PycharmProjects/cnn-rnn-master/data_set/teamGame190814/'

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

def getPage(opener,url):
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
               'Accept-Encoding':'utf-8',
               'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
               'Cache-Control': 'max-age=0',
               # 'Content - Length': '1317',
               'Content-Type': 'application/x-www-form-urlencoded',
               'Connection': 'keep-alive',
               'Host': 'www.dotamax.com',
               # 'Origin': 'http: // www.dotamax.com',
               'Upgrade-Insecure-Requests': '1',
               'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
               'Referer': 'http://www.dotamax.com/accounts/login/'}
    retryTimes = 200

    while retryTimes>0:
        try:
            timeout = 60
            socket.setdefaulttimeout(timeout)

            req = urllib.request.Request(url, headers=headers)
            response = opener.open(req)
            result = response.read().decode('UTF - 8')
            return result
        except:
            print('.')
            retryTimes-=1

def getMatchData(opener,url):
    html = getPage(opener,url)
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
    for i in range(10):  # 在页面中寻找id为playerid的 记录表现
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

def getGameList(opener,url,teamNum):
    html = getPage(opener,url)
    teamRank = [0]*8
    teamRank[0] = -1
    teamRank[1] = re.findall('700;">\d+\.+\d*%</font>', html, re.S)[0].split('>')[1].split('%')[0]
    teamRank[2] = re.findall('3627">\d*\.*\d?</font>', html, re.S)[0].split('>')[1].split('<')[0]
    gameidOne = re.findall('/match/detail/.*?" style', html, re.S)[0].split('detail/')[1].split('\'')[0]
    newUrl = "http://www.dotamax.com/match/detail/"+gameidOne
    playerIds,_,winlose=getMatchData(opener,newUrl)
    if winlose[0] == int(teamNum) or -1*winlose[0] == int(teamNum):
        teamRank[3:8] = playerIds[0:5]
    elif winlose[1] == int(teamNum) or -1*winlose[1] == int(teamNum):
        teamRank[3:8] = playerIds[5:10]
    p=1
    while p<11:
        pageUrl = url + "&p="+str(p)
        html = getPage(opener,pageUrl)
        gameids = re.findall('/match/detail/.*?" style', html, re.S)
        for gameId in gameids:
            gameid = gameId.split('detail/')[1].split('\'')[0]
            print(gameid)
            f = open(teamGame+str(teamNum) + '.txt', 'a+')
            f.write(gameid + '\n')
            f.close
        p += 1
    return teamRank

def getOpener(head):
    # deal with the Cookies
    cj = http.cookiejar.CookieJar()
    pro = urllib.request.HTTPCookieProcessor(cj)
    opener = urllib.request.build_opener(pro)
    header = []
    for key, value in head.items():
        elem = (key, value)
        header.append(elem)
    opener.addheaders = header
    return opener

def encry(message):
    rsa_e = 10001
    rsa_n = 'B81E72A33686A201B0AC009D679750990E3D168670DC6F9452C24E5A4C299AC002C6C89C3CB387' \
            '84AEA95D66B7B3E9CA950EB9EEFB4EF61383EDDB67C35727F9CA87EE3238346C66D042B3581217' \
            '9501F472AD4F3BA19E701256FE0435AB856E5C5BEA24A2387153023CD4CD43CDA7260FCC1E2E49C' \
            '14102C253F559F9A45D59DF5004A017B1239448A9A001D276CAD12535DEDE89FFBD57D75BBC9B57' \
            '5530DDD1B7FAD46064AD3C640CBD017F58981215B2EE17CBE175C36570C5235902818648577234E70E' \
            '81133B088164F98E605D0D6E69A6095A32A72' \
            '511E9AC901727B635CE2E8002A7B0EC8D012606903BCB825E60C7B6619FFCED4401E693F5EC68AB'
    rsaPublickey  = int (rsa_n,16)
    rsaPrivate=int (str(rsa_e),16)
    key = rsa.PublicKey(rsaPublickey,rsaPrivate)
    message = message.encode()
    passwd  = rsa.encrypt(message,key)
    passwd64  = binascii.b2a_base64(passwd)
    message_encrypt = bytes.decode(passwd64)
    return message_encrypt

def lgin(opener):
    headres1 = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Encoding': 'utf-8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Cache-Control': 'max-age=0',
                # 'Content - Length': '1317',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Connection': 'keep-alive',
                'Host': 'www.dotamax.com',
                # 'Origin': 'http: // www.dotamax.com',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
                'Referer': 'http://www.dotamax.com/accounts/login/'}
    # session_requests = requests.session()
    login_url = "http://www.dotamax.com/accounts/login/"
    # result = session_requests.get(login_url)
    # tree = html.fromstring(result.text)
    # authenticity_token = list(set(tree.xpath("//input[@name='csrfmiddlewaretoken']/@value")))[0]
    req_csrf = urllib.request.Request(login_url, headers=headres1)
    response_csrf = opener.open(req_csrf)
    print('Sutatue:', response_csrf.status)
    htmls = response_csrf.read().decode('utf-8')
    # csrf_pattern = re.compile("name ='csrfmiddlewaretoken' value ='.*?'")
    # authenticity_token = re.search(csrf_pattern, html).group(1)  # 获得csrf码
    tree = html.fromstring(htmls)
    authenticity_token = list(set(tree.xpath("//input[@name='csrfmiddlewaretoken']/@value")))[0]
    print(authenticity_token)

    phoneNumCipherb64 = encry('13151071096')
    passwordCipherb64 = encry('qzylove123')
    usernameCipherb64 = encry('qizhen816')
    # authenticity_token = encry(authenticity_token)

    payload = parse.urlencode({
        "csrfmiddlewaretoken": authenticity_token,
        "phoneNumCipherb64": phoneNumCipherb64,
        "usernameCipherb64": usernameCipherb64,
        "passwordCipherb64": passwordCipherb64,
        "account-type": "2",
        "src": "None"
    }).encode(encoding='UTF8')

    # result = session_requests.post(
    #     login_url,
    #     data=payload,
    #     headers=headres1)

    req = urllib.request.Request(login_url, data=payload, headers=headres1)
    response = opener.open(req)  # 至此完成登陆操作，并将cookie保存至response中。
    html_login = response.read().decode('UTF - 8')
    # print(html_login)
    return opener,payload,headres1

def makeToplist(opener):
    url="http://www.dotamax.com/match/tour_famous_team_list/?time=all"

    # opener,payload,headres1 = lgin(opener)

    # req = urllib.request.Request(url, data=payload, headers=headres1)
    # response = opener.open(req)
    # result = response.read().decode('UTF - 8')

    result = getPage(opener,url)

    teamInfo = [0]*9
    htmls = result
    teams = re.findall('team_id=.*?;',htmls,re.S)
    htmlNew = getPage(opener,"http://www.dotamax.com/match/tour_famous_team_list/?skill=&ladder=&time=all&p=2")
    team21 = re.findall('team_id=.*?;',htmlNew,re.S)[0]
    teams.append(team21)
    team22 = re.findall('team_id=.*?;',htmlNew,re.S)[1]
    teams.append(team22)

    i = 0
    for items in teams:
        teamInfo[0] = teams[i].split('=')[1].split('\'')[0]
        newUrl = "http://www.dotamax.com/match/tour_team_detail/?team_id=" + str(teamInfo[0])
        teamInfo[1:9] = getGameList(opener,newUrl,teamInfo[0])[:]
        teamInfo[1] = 21-i
        #9个数据分别为：队伍ID，名次补码，胜率，MMR，五名队员
        writeData(topList,teamInfo)
        i=i+1
    return opener

def readFirstList(opener,TL = topList, renew = False):
    Teams = []
    if not os.path.exists(TL):
        makeToplist(opener)
    with open(TL) as f:
        for line in f:
            Teams.append(line.split())
    if renew == True:
        for teamInfo in Teams:
            newUrl = "http://www.dotamax.com/match/tour_team_detail/?team_id=" + str(teamInfo[0])
            teamInfo[1:9] = getGameList(opener,newUrl,teamInfo[0])[:]
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

def getRecentGame(opener,Teams,gameID,defaultF = 0):
    print('获取数据：Game ID:',gameID)
    url = 'http://www.dotamax.com/match/detail/'+str(gameID)
    result = getMatchData(opener,url)
    if result == -1 or result == 0:
        return -1
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
        flag = defaultF
        count = 0
        errorCount = 0
        for games in teamlist:
            if flag == 1:
                print('>>>比赛方：',i,'，最近第',count,'场：',str(games[0]))
                try:
                    resu = getMatchData(opener,'http://www.dotamax.com/match/detail/'+str(games[0]))
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
                playerflag = 1
                for player in playerIdOne[j*5:j*5+5]:
                    if player not in playerIds:
                        playerflag = 0
                if playerflag!= 0:
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

def getRecentGameNoID(opener,Teams,allteams):

    team1ID,team2ID = Teams
    recentGameNum = 20

    team1list = readList(teamGame+str(team1ID)+'.txt')
    team2list = readList(teamGame+str(team2ID)+'.txt')
    teamIDs = [team1ID,team2ID]
    teamlists = [team1list,team2list]
    i = 0
    oneData = [0]*(recentGameNum*104)
    #102含义: A vs. B 的比赛：
    # A1*10 A2*10 A3*10 A4*10 A5*10 WinLose anotherTeam B1*10 B2*10 B3*10 B4*10 B5*10 WinLose anotherTeam 共104
    # 以上的数据*要收集的比赛场数
    teamData = []

    for team in allteams:
        if team[0] == team1ID:
            teamData.append(team)
        if team[0] == team2ID:
            teamData.append(team)

    for teamlist in teamlists:
        flag = 1
        count = 0
        for games in teamlist:
            if flag == 1:
                print('>>>比赛方：',i,'，最近第',count,'场：',str(games[0]))
                try:
                    resu = getMatchData(opener,'http://www.dotamax.com/match/detail/'+str(games[0]))
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


                oneData[count*104+i*52:count*104+i*52+50] = gameDataOne[j*50:j*50+50]
                oneData[count * 104 + i * 52+50] = win
                oneData[count * 104 + i * 52+51] = anotherTeam
                    #data[i].append(gameDataOne[j*5:j*5+5]+win)
                count+=1

            if count == recentGameNum:
                break

        i+=1
        if count!=recentGameNum:
            print('数据不足')
        if count<10:
            print('Error:数据不足')
            return  -1

    print('*一条数据：')
    print(len(oneData))
    print(oneData)
    return oneData,teamData

def getStart(opener):
    opener,payload,headres1 = lgin(opener)

    Teams,Visit = readFirstList(opener,renew=True)
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
                    resu = getRecentGame(opener,Teams,newID)
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

def newop():
    cookie_filename = 'cookie.txt'
    fobj = open(cookie_filename,'w')
    fobj.close()  # 创建一个txt储存cookie信息，便于下次登陆使用，该项可以忽略
    cookiejar = http.cookiejar.LWPCookieJar(cookie_filename)
    handler = urllib.request.HTTPCookieProcessor(cookiejar)
    opener = urllib.request.build_opener(handler)
    return opener

opener = newop()
# opener = makeToplist(opener)
# getStart(opener)
