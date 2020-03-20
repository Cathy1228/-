import pandas as pd
import numpy as np
pd.read_table('meal_order_info.csv', encoding='gbk', sep=',')
data=pd.read_table('meal_order_detail.csv', sep=',')
data['dishes_name']=data['dishes_name'].map(lambda x : x.replace('\r\n',''))#处理字符数据
data['emp_id']=data['emp_id'].astype(np.int)
data_1=data[['dishes_name','counts']].groupby(by='dishes_name')
data_2=data_1.sum()

def MinMaxScale(data_2):
    return(data_2-data_2.min())/(data_2.max()-data_2.min())
a=MinMaxScale(data_2['counts'])
df=pd.DataFrame(a)
data_3=df.sort_values('counts',ascending=False)
top_10=data_3.iloc[0:10, 0:1]
print(top_10)
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1)
top_10.plot.bar(ax=axes[0],color='k',alpha=0.7,rot=0)

# 参数alpha指定了所绘制图形的透明度，rot指定类别标签偏转的角度
 
top_10.plot.barh(ax=axes[1],color='k',alpha=0.7)
# Series.plot.barh()的用法与Series.plot.bar()一样，只不过绘制的条形图是水平方向的
fig.savefig('p1.png')
plt.show()#top10条形图

info=pd.read_table('meal_order_info.csv', encoding='gbk', sep=',')
a=info["order_status"].value_counts()/info['order_status'].count()
print(a)#求百分比

data_4=data[['dishes_name','emp_id']]
data_4['emp_id']=data_4['emp_id'].astype(np.int)
print(data_4)#提取主要特征

print(data)
print(info)

info_did=pd.DataFrame.drop_duplicates(info, subset='info_id', keep='first', inplace=False)
print(info_did)#去重，看订单个数

data_4=data[['order_id','amounts']].groupby(by='order_id')
data_5=data_4.sum()
print(data_5)#各订单对应的销售金额


data_6=data.loc[data['order_id'] > 1000, :]
print(data_6['order_id'])
import numpy as np

data_did=data[['dishes_name','emp_id']]
from sklearn.model_selection import train_test_split

train_empid, test_empid,train_dishes,test_dishes = train_test_split(data_did['emp_id'], data_did['dishes_name'], test_size=0.2)
print(train_empid)
print(test_empid)#以客户id划分训练集测试集

#info_did['info_id']=train_infoid
#info_train=info_did[info_did['info_id'].notna()]#info表的以info_id划分的训练集
#info_did['info_id']=test_infoid
#info_test=info_did[info_did['info_id'].notna()]#info表的以info_id划分的测试集
data_did['emp_id']=train_empid
emp_train=data_did[data_did['emp_id'].notna()]#data4表的以emp_id划分的训练集  但是在原表修改了
emp_train['emp_id']=emp_train['emp_id'].astype(np.int)
data_did['emp_id']=test_empid
emp_test=data_did[data_did['emp_id'].notna()]#info表的以emp_id划分的测试集
emp_test['emp_id']=emp_test['emp_id'].astype(np.int)
print(emp_train)
print(emp_test)


#info_did['emp_id'].dropna(inplace=False)
#emp_train=info_did
#print(emp_train)
#info_doup=emp_train[emp_train['dishes_count']>3]
#print(info_doup)#保留点菜数为3以上的客户
import numpy as np
emp_train['value']=1#增添一列
#emp_train['emp_id'].astype(np.int)
print('emp',emp_train)

#菜品相似度

emp_train2=emp_train[['emp_id','dishes_name','value']].groupby(by=['emp_id','dishes_name'])
emp_train3=emp_train2.sum()
print(emp_train3)

emp_train
juzhen=pd.pivot_table(emp_train, values='value', index='emp_id', columns='dishes_name', aggfunc='count', margins=False, dropna=True, margins_name='All')
erwei=juzhen.fillna(value=0)
print(erwei)#客户-菜名的0/1二维矩阵
print(type(erwei))
print(erwei.iloc[[0,2],[1,2]])
print(erwei.iloc[[1,2],[1,2]])
print(erwei.describe())
print(erwei[erwei.iloc[:,[0,1]]>0])
print(erwei.columns)
print(erwei[' 42度海之蓝'])
print(erwei[erwei[' 42度海之蓝']>0])

#菜品相似度1
cou1=erwei[erwei[' 42度海之蓝']>0]
print(len(cou1))
cou2=erwei[erwei[' 北冰洋汽水 ']>0]
print(len(cou2))
and12=erwei[(erwei[' 42度海之蓝']>0)&(erwei[' 北冰洋汽水 ']>0)]
print(len(and12))
lv=len(and12)/(len(cou1)+len(cou2)-len(and12))
print(lv)
#菜品相似度2
cou11=erwei[erwei.columns[0]].astype(np.int)
cou111=cou11[cou11>0]
print(len(cou111))
cou22=erwei[erwei.columns[1]].astype(np.int)
cou222=cou22[cou22>0]
print(len(cou222))
and1212=erwei[(cou11>0)&(cou22>0)]
print(len(and1212))
lvlv=len(and1212)/(len(cou111)+len(cou222)-len(and1212))
print(lvlv)

for i in range(erwei.shape[1]):
    cou11 = erwei[erwei.columns[i]].astype(np.int)
    cou111 = cou11[cou11 > 0]
    #print(len(cou111))

    for j in range(i, erwei.shape[1]):
        cou22 = erwei[erwei.columns[j]].astype(np.int)
        cou222 = cou22[cou22 > 0]
        #print(len(cou222))

        and1212 = erwei[(cou11 > 0) & (cou22 > 0)]
        #print(len(and1212))
        lvlv = len(and1212) / (len(cou111) + len(cou222) - len(and1212))
        print(i,j,lvlv)


"""
#矩阵
similar=np.array((erwei.shape[1],erwei.shape[1]),dtype=np.float32)

for i in range(erwei.shape[1]):
    a=erwei[erwei.columns[i]].astype(np.int)
    A=a[a>0]
    #print(len(A))

    for j in range(i,erwei.shape[1]):
        b=erwei[erwei.columns[j]].astype(np.int)
        B=b[b>0]
        #print(len(B))
        A_with_B=erwei[(a>0)&(b>0)]
        #print(len(A_with_B))
        s=len(A_with_B)/(len(A)+len(B)-len(A_with_B))
        if i==j:
           s-=1
        similar[i][j]=(s if i!=j else s-1)
  #print(similar)
"""

