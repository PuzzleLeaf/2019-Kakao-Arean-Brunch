
# coding: utf-8

# In[ ]:


from collections import Counter
from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re
import time
import sys
import numpy as np
import pandas as pd
import math
import copy
from time import gmtime, strftime

# 설정 파일
import config as conf


# In[ ]:


def drawProgressBar(percent, barLen = 20):
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()


# In[ ]:


print("Start")


# In[ ]:


# 예측 유저 목록
predict_users = pd.read_csv(conf.predict_res, names=['user_id'])


# In[ ]:


# 메타 데이터
metadata = pd.read_json('res/metadata.json', lines=True)


# In[ ]:


after_oct_articles = []
after_oct_articles = metadata[metadata.reg_ts >= 1538352000000]


# In[ ]:


# 각 매거진 별로 얼마나 글이 있는지 체크해보자
mag_ids = list(after_oct_articles.magazine_id)

u_mag_ids = set(mag_ids)

mag_count_dic = {}
for m_id in u_mag_ids:
    mag_count_dic[m_id] = mag_ids.count(m_id)


# In[ ]:


# 유저 데이터
users = pd.read_json('res/users.json', lines=True)
users = users[['following_list', 'id', 'keyword_list']]


# In[ ]:


# read 데이터
read_file_lst = glob.glob('res/read/*')
exclude_file_lst = ['read.tar']

read_df_lst = []
for f in read_file_lst:
    file_name = os.path.basename(f)
    if file_name in exclude_file_lst:
        print(file_name)
    else:
        df_temp = pd.read_csv(f, header=None, names=['raw'])
        df_temp['dt'] = file_name[:8]
        df_temp['hr'] = file_name[8:10]
        df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0]
        df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip()
        read_df_lst.append(df_temp)
        
read = pd.concat(read_df_lst)


# In[ ]:


# read 데이터 가공
def chainer(s):
    return list(chain.from_iterable(s.str.split(' ')))
read_cnt_by_user = read['article_id'].str.split(' ').map(len)
read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                         'hr': np.repeat(read['hr'], read_cnt_by_user),
                         'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                         'article_id': chainer(read['article_id'])})
read_raw = read_raw[['dt', 'hr', 'user_id', 'article_id']]


# In[ ]:


# 글별 소비수 통계
atc_read_cnt = read_raw[read_raw.article_id != ''].groupby('article_id')['user_id'].count()
atc_read_cnt = atc_read_cnt.reset_index()
atc_read_cnt.columns = ['article_id', 'read_cnt']


# In[ ]:


author_read_dic = {}

for index, row in atc_read_cnt.iterrows():
    atc = row['article_id']
    author = ((row['article_id']).split("_"))[0]
    if author in author_read_dic:
        author_read_dic[author].append(atc)
    else:
        article_list = []
        article_list.append(atc)
        author_read_dic[author] = article_list


# In[ ]:


# 윈도우 예외처리 추가
atc = metadata.copy()
atc['reg_datetime'] = atc['reg_ts'].apply(lambda x :datetime.fromtimestamp(x/1000.0) if x/1000.0 != 0.0 else datetime(1970, 1, 1, 0, 0))
atc.loc[atc['reg_datetime'] == atc['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)
atc['reg_dt'] = atc['reg_datetime'].dt.date
atc['type'] = atc['magazine_id'].apply(lambda x : '개인' if x == 0.0 else '매거진')
# 컬럼명 변경
atc.columns = ['id', 'display_url', 'article_id', 'keyword_list', 'magazine_id', 'reg_ts', 'sub_title', 'title', 'author_id', 'reg_datetime', 'reg_dt', 'type']


# In[ ]:


#metadata 결합
atc_read_cnt = pd.merge(atc_read_cnt, atc, how='left', left_on='article_id', right_on='article_id')
atc_read_cnt_nn = atc_read_cnt[atc_read_cnt['id'].notnull()]


# In[ ]:


read_cnt_frame = atc_read_cnt_nn.sort_values(["read_cnt"], ascending=[False])
optimize_frame = read_cnt_frame.drop(['id', 'display_url', 'sub_title', 'magazine_id', 'reg_ts', 'title', 'author_id', 'reg_dt', 'type'], axis=1)


# In[ ]:


# article 기반 데이터 정제
article_detail_dic = {}
for row in optimize_frame.values:
    article_detail_dic[row[0]] = {'read_cnt': row[1], 'keyword': row[2] ,'datetime': row[3]}


# In[ ]:


# 전체 유저별 읽은 글 목록
user_read_dic = {}
for row in read_raw.values:
    user_id = row[2]
    article = row[3]
    if user_read_dic.get(user_id, "empty") == "empty":
        user_read_dic[user_id] = [article]
    else:
        user_read_dic[user_id].append(article)


# In[ ]:


#유저별 팔로우 리스트
user_follow_list = []
for row in users.values:
    user_id = row[1]
    follow_list = row[0]
    if "@brunch" in follow_list: follow_list.remove("@brunch")

    if len(follow_list) > 1:
        user_follow_list.append(follow_list)


# In[ ]:


# 유저별 팔로우 목록
user_follow_dict = {}
for row in users.values:
    user_id = row[1]
    follow_list = row[0]
    user_follow_dict[user_id] = follow_list


# In[ ]:


# 유저의 follow가 아니면서 읽은 글 수가 많은 작가 추리기 : 전체 유저가 읽은 목록 대상 조회
followable_dic = {}

for user_id, value in user_read_dic.items():
    arcs = list(value).copy()
    authors_map = map(lambda x: (x.split("_"))[0], arcs)
    authors = list(authors_map).copy()
    
    orinC1 = set(authors)
    C1 = list(orinC1).copy()
    
    dic = {}
    for at in C1:
        dic[at] = authors.count(at)
    
    five_percent = round(len(user_read_dic[user_id]) / 5)
    
    for k, v in dic.items():
        if v > 10: #or v > five_percent:
            # 팔로우하는 작가가 있는 유저
            if user_id in user_follow_dict:
                if not k in (user_follow_dict[user_id]):
                    if user_id in followable_dic:
                        followable_dic[user_id].append(k)
                    else:
                        auth_list = []
                        auth_list.append(k)
                        followable_dic[user_id] = auth_list
            else:
                if user_id in followable_dic:
                    followable_dic[user_id].append(k)
                else:
                    auth_list = []
                    auth_list.append(k)
                    followable_dic[user_id] = auth_list


# In[ ]:


def convertTime (dateTime):
    if dateTime == 0:
        return 0
    t = pd.Timestamp(dateTime)
    return time.mktime(t.timetuple())


# In[ ]:


user_last_read_dic = {}
for row in read_raw.values:
    if user_last_read_dic.get(row[2], "empty") == "empty":
        user_last_read_dic[row[2]] = {}
    user_last_read_dic[row[2]][row[3]] = row[0]


# In[ ]:


# followable 얻기위해 apriori 알고리즘 사용
print("[Train-1] Start")
test_list = user_follow_list.copy()

count = len(test_list)

def createC1(dataSet):
    C1=[]
    
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    # 요소 n 은 몇개의 그룹에 포함되어 있는지 계산한다.
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can] = 1 
                else: ssCnt[can] += 1

    
    # 지지도가 0.5보다 높은것들의 리스트를 구함. 
    numItems = float(count) # 그룹 갯수
    retList = []
    supportData = {}
    for key , value in ssCnt.items():
        try:
            support = value / numItems
            
            if support >= minSupport:
                retList.insert(0,key)
            
            supportData[key] = support
        except ZeroDivisionError:
            print('zero_division_error')
        
        

    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])

    return retList

# 특정 지지도 이상의 값들의 쌍을 찾음 
def apriori(dataset, minSupport = 0.01):
    orinC1 = createC1(dataset)
    C1 = list(orinC1).copy()
    
    orinD = map(set, dataset)
    D = list(orinD).copy()

    L1 , supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport) # 후보그룹을 모두 찾는다.
        supportData.update(supK)
        L.append(Lk) #이게 핵심!특정 지지도 이상의 그룹들만 L에 담는다.즉 가지치기
        k += 1
        
    return L, supportData

def generateRules(L, supportData, minConf=0.01):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i>1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet,H1,supportData, bigRuleList, minConf)

    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.01):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf >= minConf:
            #print (freqSet-conseq, '-->', conseq, 'conf:', conf)
            
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.01):
    m = len(H[0])
    
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)

        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)
            
L, suppData = apriori(test_list, minSupport = 0.01)
# print ("L:" + str(L))
# print (".........................")
# print ("suppData:" + str(suppData))
rules = generateRules(L, suppData, minConf=0.01)


# In[ ]:


relative_dic = {}
for rule in rules:
    conf = rule[2]
    if conf > 0.3:
        user = list(rule[0])[0]
        target = list(rule[1])[0]
        if user in relative_dic:
            relative_dic[user].append(target)
        else:
            target_list = []
            target_list.append(target)
            relative_dic[user] = target_list


# In[ ]:


# followable_dic 목록에 연관성이 30%이상인 팔로워 목록을 추가한다
step = 0
step_end = len(user_follow_dict.items())
for key, follower_list in user_follow_dict.items():
    step +=1
    drawProgressBar(step / step_end)
    avail = []
    
    for follower in follower_list:
        if follower in relative_dic:
            avail.extend(relative_dic[follower])
    
    if len(avail) > 0:
        #print(str(avail))
        for author in avail:
            if not author in follower_list:
                if not key in followable_dic:
                    #print('make new list: ' + key)
                    new_list = []
                    followable_dic[key] = new_list
                    
                followable_dic[key].append(author)
                #print('added \'' + author + '\' in ' + key)


# In[ ]:


# 키워드 목록
keyword_list = list(after_oct_articles.keyword_list)
ids = list(after_oct_articles.id)
size = len(after_oct_articles)

key_dic = {}
for i in range(0, size):
    if len(keyword_list[i]) > 0:
        key_dic[ids[i]] = keyword_list[i]


# In[ ]:


# 많이 읽은 키워드 정리
step = 0
step_end = len(user_read_dic.items())

user_keywords_dic = {}

for user_id, value in user_read_dic.items():
    step +=1
    drawProgressBar(step / step_end)
    
    arcs = list(value).copy()
    has_key_cnt = 0
    dic = {}
    for arc in arcs:
        # 매거진 글인지 체크
        if arc in key_dic:
            has_key_cnt += 1 # 키워드가 있는 글을 몇 개나 읽었나
            key_list = key_dic[arc]
            
            for key in key_list:
                if not key in dic:
                    dic[key] = 0
                dic[key] += 1 # 읽은 글의 키워드 카운팅
    
    for k, v in dic.items():
        if v < 10: continue
        score = v / has_key_cnt
        # print(k + ' : ' + str(score))
        
        if not user_id in user_keywords_dic:
            score_dic = {}
            user_keywords_dic[user_id] = score_dic
        
        (user_keywords_dic[user_id])[k] = score


# In[ ]:


# 매거진 정리
has_mag_atcs = after_oct_articles[after_oct_articles.magazine_id > 0]


# In[ ]:


ids = list(has_mag_atcs.id)
mags = list(has_mag_atcs.magazine_id)

size = len(has_mag_atcs)
mags_dic = {}

for i in range(0, size):
    mags_dic[ids[i]] = mags[i]


# In[ ]:


# 유저의 매거진 글 선호도 체크
user_mag_dic = {} # dic[dic[int]] 객체

step = 0
step_end = len(user_read_dic.items())

for user_id, value in user_read_dic.items():
    step +=1
    drawProgressBar(step / step_end)
    
    arcs = list(value).copy()
    
    dic = {}
    for arc in arcs:
        # 매거진 글인지 체크
        if arc in mags_dic:
            mag_id = mags_dic[arc]
            if not mag_id in dic:
                dic[mag_id] = 0
            dic[mag_id] += 1 # 읽은 매거진 글 카운팅
    
    for k, v in dic.items():
        arc_cn = mag_count_dic[k]
        prefer = v / arc_cn
        # print('arc_cn: ' + str(arc_cn) + ': ' + str(v))
        
        if not user_id in user_mag_dic:
            prefer_dic = {}
            user_mag_dic[user_id] = prefer_dic
            
        # 여러번 읽은 경우에 최대치 1
        if prefer > 1.0:
            prefer = 1.0
        (user_mag_dic[user_id])[k] = prefer


# In[ ]:


# 유저의 작가별 글 선호도 체크
user_prefer_dic = {} # dic[dic[float]] 객체

step = 0
step_end = len(user_read_dic.items())

for user_id, value in user_read_dic.items():
    step +=1
    drawProgressBar(step / step_end)
    
    arcs = list(value).copy()
    authors_map = map(lambda x: (x.split("_"))[0], arcs)
    authors = list(authors_map).copy()
    
    orinC1 = set(authors)
    C1 = list(orinC1).copy()
    
    dic = {}
    for at in C1:
        if len(at) > 0:
            dic[at] = authors.count(at)
    #print(str(dic))
    
    for k, v in dic.items():
        prefer = v / len(author_read_dic[k]) # 작가가 쓴 글에서 읽은 글 수
        
        if user_id in user_prefer_dic:
            (user_prefer_dic[user_id])[k] = prefer
        else:
            prefer_dic = {}
            prefer_dic[k] = prefer
            user_prefer_dic[user_id] = prefer_dic


# In[ ]:


# 포인트 계산 - 이게 오래 걸림

temp = {}
ids = list(after_oct_articles.id)
users = list(after_oct_articles.user_id)
times = list(after_oct_articles.reg_ts)
size = len(after_oct_articles)

for i in range(0, size):
    temp[ids[i]] = {"author": users[i], "ts": times[i]}
    
#print(len(temp))

all_read_cnt = len(read)
#print(all_read_cnt)
maxT = 1554044340000
term = maxT - 1538352219000 # 1546300963000
term2 = maxT - 1548979335000 # 2월 이후 글 중에서 

pre_atc_dic = {}
feb_articles_dic = {}

step = 0
step_end = len(temp.items())

for key, value in temp.items():
    step +=1
    drawProgressBar(step / step_end)
    
    author = value['author']
    arc_id = key
    popular = 0
    read_cnt = 0
    
    if (atc_read_cnt['article_id'] == arc_id).any():
        a = atc_read_cnt[atc_read_cnt['article_id'] == arc_id]['read_cnt']
        read_cnt = a.values[0]
        if read_cnt > 10:
            popular = (read_cnt / all_read_cnt) * 1000 # 글의 유명세
            # print(popular)
    
    add_dic = {}
    date = (maxT - value['ts']) / term # 최신에 가까울수록 1

    date2 = 0
    if value['ts'] > 1548979335000: # 2월 이후 애들끼리 경쟁
        date2 = (maxT - value['ts']) / term2 
    
    # 옛날 글 일수록 점수 많이 낮아지라고..
    if date >= 0.1:
        date = math.log10(date * 10)
   
    add_dic['date_point'] = date
    add_dic['popular'] = popular
    add_dic['author'] = author
    # print(str(add_dic))
    
    if value['ts'] > 1548979335000:
        add_dic['date_point'] = date2
        feb_articles_dic[arc_id] = add_dic # 2월 이후 글
    else:
        pre_atc_dic[arc_id] = add_dic # 2월 이전 전체 글


# In[ ]:


result_dic = {}
step = 0
step_end = len(predict_users.user_id)

for user in predict_users.user_id:
    step +=1
    drawProgressBar(step / step_end)
    
    arc_dic = {}
   
    is_in_preferdic = False
    if user in user_prefer_dic:
        is_in_preferdic = True
        
    has_follow = False
    if user in user_follow_dict:
        has_follow = True
    
    has_followable = False
    if user in followable_dic:
        has_followable = True
        
    has_mag_dic = False
    if user in user_mag_dic:
        has_mag_dic = True
        
    has_key_dic = False
    if user in user_keywords_dic:
        has_key_dic = True
        
    for key, value in feb_articles_dic.items():
        # print(value)
        author = value['author']
        arc_id = key
        date = value['date_point']
        popular = value['popular']
        
        # 유저가 이미 읽은글인지 판단
        if arc_id in user_read_dic[user]:
            # print('already read')
            continue
        
        au_prefer = 0.0
        if is_in_preferdic:
            if author in user_prefer_dic[user]:
                au_prefer = user_prefer_dic[user][author] # 작가 선호도
        
        mag_prefer = 0.0
        if has_mag_dic:
            if arc_id in mags_dic:
                mag_id = mags_dic[arc_id]
                if mag_id in user_mag_dic[user]:
                    mag_prefer = user_mag_dic[user][mag_id] # 매거진 선호도
        
        keyword_score = 0.0

        if has_key_dic and arc_id in key_dic:
            keys = key_dic[arc_id]
            # print('key_list: ' + str(keys))
            
            for w in keys:
                if w in user_keywords_dic[user]:
                    keyword_score += user_keywords_dic[user][w]
        
        if keyword_score > 1.0:
            keyword_score = 1.0

        base = date + au_prefer + popular + mag_prefer + keyword_score
  
        # 팔로우 작가가 있는 유저
        if has_follow:
            if author in (user_follow_dict[user]): # 팔로우 작가면
                base = base * 3
                # print("follow author " + str(base))
                
        if has_followable:
            # print("followable" + followable_dic[user])
            if author in (followable_dic[user]): # 글 많이 읽은 작가면
                base = base * 1.5
                # print("followable author " + str(base))
        
        if base > 1.8:
            # print("--- id: " + arc_id + " : " + str(base))
            arc_dic[arc_id] = base
    
    arc_dic = sorted(arc_dic, key = lambda k : arc_dic[k], reverse = True)
    # print(str(arc_dic[:100]))
    
    arc_list = list(arc_dic).copy()
    # print(str(len(arc_list)))
    if len(arc_list) >= 100:
        result_dic[user] = arc_list[:100]
        # print('all list fill')
    else: 
        remain = 100 - len(arc_list)
        result_dic[user] = arc_list
        # print('remain the list: ' + str(remain))


# In[ ]:


# 2차 학습
print("[Train-2] Start")
train2_data = atc_read_cnt_nn.sort_values(["read_cnt"], ascending=[False]).article_id


# In[ ]:


# 중복없이 100개 추출하기
step = 0
step_end = len(predict_users.user_id)
temp_result = {}
result = {}

predict_ids = predict_users.user_id
already_use = []

for id in predict_users.user_id:
    step +=1
    drawProgressBar(step / step_end)
    
    result[id] = result_dic[id]# merge_dic[id][:35]
    already_use.extend(result[id])
    
    train2_data = [article for article in train2_data if article not in already_use]
    already_use = []
    
    remain_cnt = 100 - len(result_dic[id])
    temp_result[id] = train2_data[:remain_cnt]
    already_use.extend(temp_result[id])
    
    result[id].extend(temp_result[id])
    if len(result[id]) != 100:
        print("Error")


# In[ ]:


print("[Train-2] End")
print("All Train Success")


# In[ ]:


# 저장하기
save_data = []
for idx in range(0, len(predict_users.user_id)):
    user_id = predict_users.user_id[idx]
    temp = [user_id]
    temp.extend(result[user_id])
    temp = [' '.join(temp)]
    save_data.append(temp)


# In[ ]:


save = pd.DataFrame(save_data)


# In[ ]:


save.to_csv("result.csv", header=False, index=False);
print("[Success] result.csv")

