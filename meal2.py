from math import sqrt
import operator

#1.构建用户-->物品的倒排
def loadData(files):
    data ={};
    for line in files:
        user,score,item=line.split(",");
        data.setdefault(user,{});
        data[user][item]=score;
    print("----1.用户：物品的倒排----")
    print (data)
    return data

#2.计算
# 2.1 构造物品-->物品的共现矩阵
# 2.2 计算物品与物品的相似矩阵
def similarity(data):
    # 2.1 构造物品：物品的共现矩阵
    N={};#喜欢物品i的总人数
    C={};#喜欢物品i也喜欢物品j的人数
    for user,item in data.items():
        for i,score in item.items():
            N.setdefault(i,0);
            N[i]+=1;
            C.setdefault(i,{});
            for j,scores in item.items():
                if j not in i:
                    C[i].setdefault(j,0);
                    C[i][j]+=1;

    print ("---2.构造的共现矩阵---")
    print ('N:',N);
    print ('C',C);

    #2.2 计算物品与物品的相似矩阵
    W={};
    for i,item in C.items():
        W.setdefault(i,{});
        for j,item2 in item.items():
            W[i].setdefault(j,0);
            W[i][j]=C[i][j]/sqrt(N[i]*N[j]);
    print ("---3.构造的相似矩阵---")
    print (W)
    return W

#3.根据用户的历史记录，给用户推荐物品
def recommandList(data,W,user,k=3,N=10):
    rank={};
    for i,score in data[user].items():#获得用户user历史记录，如A用户的历史记录为{'a': '1', 'b': '1', 'd': '1'}
        for j,w in sorted(W[i].items(),key=operator.itemgetter(1),reverse=True)[0:k]:#获得与物品i相似的k个物品
            if j not in data[user].keys():#该相似的物品不在用户user的记录里
                rank.setdefault(j,0);
                rank[j]+=float(score) * w;

    print("---4.推荐----")
    print(sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N]);
    return sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N];
import cx_Oracle
import csv
import re
import codecs
import requests
from bs4 import BeautifulSoup
def fetch_data(sql):
    conn = cx_Oracle.connect('C##CHINA_GOOD/bishe@127.0.0.1:1521/ORCL')
    cursor = conn.cursor()
    result = cursor.execute(sql)
    all_data = cursor.fetchall()  # 查询全部
    return all_data

if __name__=='__main__':
    #用户，兴趣度，物品
    sql1 = "select * from user_like_good"
    datas = fetch_data(sql1)
    uid_score_bid=[]
    for data in datas:
        content=str(data[1])+",1,"+str(data[2])
        print(content)
        uid_score_bid.append(content)
    print(uid_score_bid)
    data=loadData(uid_score_bid);#获得数据
    W=similarity(data);#计算物品相似矩阵
    recommandList(data,W,'104976',3,10);#推荐