# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import dateutil
# print("=======================================================")
# print("groupby_hierarchical_index")
# print("=======================================================")
# # groupby
# # SQL groupby 명령어와 같음
# # split -> apply -> combine 과정을 거쳐 연산
# ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
#          'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
#          'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
#          'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
#          'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
# df=pd.DataFrame(ipl_data)
# print(df)
# print("\n")
# print(df.groupby("Team")["Points"].sum())
# #          groupby(묶음기준이되는column)[적용받을column].적용받는연산
# print("\n=======================================================\n")
#
# h_index=df.groupby(["Team", "Year"])["Points"].sum() # 한 개 이상의 column을 묶을 수 있음
# print(h_index)
# print("\n")
# # Hierarchical index
# # groupby 명령의 결과물도 dataframe 이다
# # 두 개의 column으로 groupby를 할 경우, index가 두 개 생성된다.
# print(h_index.index)
# print("\n")
# print(h_index["Devils":"Kings"])
# print("\n")
# # Hierarchical index - unstack()
# # Group으로 묶여진 데이터를 matrix 형태로 전환해줌
# print(h_index.unstack())
# print("\n")
# print(h_index.swaplevel())
# print("\n")
# # print(h_index.swaplevel().sortlevel(0))
# print("\n")
# # index level 별로 연산 가능하다
# print(h_index.sum(level=0))
# print("\n")
# print(h_index.sum(level=1))
# print("\n=======================================================\n")
# # groupby - grouped
# # Groupby에 의해 Split된 상태를 추출 가능하다.
# grouped=df.groupby("Team")
# for name, group in grouped:
#     print(type(name))
#     print(type(group))
# print("\n")
# print(grouped.get_group("Devils"))
# #    Points  Rank    Team  Year
# # 2     863     2  Devils  2014
# # 3     673     3  Devils  2015
# print("\n")
# # grouped
# # 추출된 group 정보에는 세 가지 유형의 apply가 가능하다
# # Aggregation : 요약된 통계정보를 추출해 줌
# # Transformation : 해당 정보를 변환해줌
# # Filtration : 특정 정보를 제거하여 보여주는 필터링 기능
# print(grouped.agg(min))
# print("\n")
# print(grouped.agg(np.mean))
# print("\n")
# print(grouped['Points'].agg([np.sum, np.mean, np.std]))
# # Point라는 column을 기준으로 aggregation함
# print("\n=======================================================\n")
# score=lambda x: (x.max())
# print(grouped.transform(score))
# print("\n")
# score=lambda x: (x-x.mean())/ x.std() # 정규분포
# print(grouped.transform(score))
# print("\n")
# # 특정 조건으로 데이터 검색할 때 사용
# print(df.groupby('Team').filter(lambda x: len(x) >= 3))
# # 같은 Team명의 데이터가 3개 이상인 경우 출력
# print("\n")
# # 각 그룹의 Points값이 800 이상인 그룹들만 출력
# print(df.groupby('Team').filter(lambda x: x["Points"].max() > 800))
# print("\n")
# df_phone = pd.read_csv("./data/phone_data.csv")
# print(df_phone.head())
# print("\n")
# df_phone['date']=df_phone['date'].apply(dateutil.parser.parse, dayfirst=True)
# print(df_phone.head())
# print("\n")
# print(df_phone.groupby('month')['duration'].sum())
# print("\n")
# # item이 call인 경우에만 groupby를 사용
# print(df_phone[df_phone['item']=='call'].groupby('month')['duration'].sum())
# print("\n")
# print(df_phone.groupby(['month', 'item'])['duration'].sum())
# print("\n")
# print(df_phone.groupby(['month', 'item'])['date'].count().unstack())
# print("\n")
# print(df_phone.groupby('month', as_index=False).agg({"duration":"sum"}))
# print("\n")
# print(df_phone.groupby(['month', 'item']).agg({'duration':sum, # find the sum of the durations for each group
#     'network_type':"count", # find the number of network type enties
#     'date':'first'}))  # get the first date per group
# print("\n")
# print(df_phone.groupby(['month', 'item']).agg({'duration':[min], # find the min, max, and sum of the duration column
#     'network_type':"count", # find the number of network type entries
#     'date':[min, 'first', 'nunique']})) # get the min, first, and number of unique dates
# print("\n")
# grouped=df_phone.groupby('month').agg({"duration": [min,max, np.mean]})
# print(grouped)
# print("\n")
# grouped.columns=grouped.columns.droplevel(level=0)
# print(grouped)
# print("\n")
# print(grouped.rename(columns={"min":"min_duration", "max":"max_duration", "mean":"mean_duration"}))
# print("\n")
# grouped=df_phone.groupby('month').agg({"duration":[min,max,np.mean]})
# print(grouped)
# print("\n")
# grouped.columns=grouped.columns.droplevel(level=0)
# print(grouped)
# print("\n")
# print(grouped.add_prefix("duration_"))
# print("\n")
# print(df_phone)
#
# print("\n=======================================================")
# print("pivot_crosstab")
# print("=======================================================")
# df_phone = pd.read_csv("./data/phone_data.csv")
# df_phone['date']=df_phone['date'].apply(dateutil.parser.parse, dayfirst=True)
# print(df_phone.head())
# print("\n")
# # duration이라는 데이터를 사용하여,
# # Hierarchical index로 month와 item을 쓴다.
# # column은 network별로 만들어서 쓰고, Aggregation은 sum
# # NaN값은 0으로 표시
# print(df_phone.pivot_table(["duration"], index=[df_phone.month, df_phone.item],
#     columns=df_phone.network, aggfunc="sum", fill_value=0))
# #              duration
# # network        Meteor Tesco  Three Vodafone      data landline special voicemail world
# # month   item
# # 2014-11 call     1521  4045  12458     4316     0.000     2906       0       301     0
# #         data        0     0      0        0   998.441        0       0         0     0
# #         sms        10     3     25       55     0.000        0       1         0     0
# # 2014-12 call     2010  1819   6316     1302     0.000     1424       0       690     0
# #         data        0     0      0        0  1032.870        0       0         0     0
# #         sms        12     1     13       18     0.000        0       0         0     4
# # 2015-01 call     2207  2904   6445     3626     0.000     1603       0       285     0
# #         data        0     0      0        0  1067.299        0       0         0     0
# #         sms        10     3     33       40     0.000        0       0         0     0
# # 2015-02 call     1188  4087   6279     1864     0.000      730       0       268     0
# #         data        0     0      0        0  1067.299        0       0         0     0
# #         sms         1     2     11       23     0.000        0       2         0     0
# # 2015-03 call      274   973   4966     3513     0.000    11770       0       231     0
# #         data        0     0      0        0   998.441        0       0         0     0
# #         sms         0     4      5       13     0.000        0       0         0     3
# print("\n=======================================================\n")
#
# # pivot_crosstab
# df_movie = pd.read_csv("./data/movie_rating.csv")
# print(df_movie.head())
# print("\n")
# # index(세로)에는 critic을 넣어주고, column(가로)에는 title을 넣어준다.
# # values는 rating값들을 넣고, aggreagtion은 first로 함(sum이나 등등도 가능)
# print(pd.crosstab(index=df_movie.critic,columns=df_movie.title,values=df_movie.rating,
#     aggfunc="first").fillna(0))
# # title          Just My Luck  Lady in the Water  Snakes on a Plane  Superman Returns  The Night Listener  You Me and Dupree
# # critic
# # Claudia Puig            3.0                0.0                3.5               4.0                 4.5                2.5
# # Gene Seymour            1.5                3.0                3.5               5.0                 3.0                3.5
# # Jack Matthews           0.0                3.0                4.0               5.0                 3.0                3.5
# # Lisa Rose               3.0                2.5                3.5               3.5                 3.0                2.5
# # Mick LaSalle            2.0                3.0                4.0               3.0                 3.0                2.0
# # Toby                    0.0                0.0                4.5               4.0                 0.0                1.0
# print("\n")
# print(df_movie.pivot_table(["rating"], index=df_movie.critic, columns=df_movie.title,
#     aggfunc="sum", fill_value=0)) # pivot table로도 가능하다.
# #                     rating
# # title         Just My Luck Lady in the Water Snakes on a Plane Superman Returns The Night Listener You Me and Dupree
# # critic
# # Claudia Puig           3.0               0.0               3.5              4.0                4.5               2.5
# # Gene Seymour           1.5               3.0               3.5              5.0                3.0               3.5
# # Jack Matthews          0.0               3.0               4.0              5.0                3.0               3.5
# # Lisa Rose              3.0               2.5               3.5              3.5                3.0               2.5
# # Mick LaSalle           2.0               3.0               4.0              3.0                3.0               2.0
# # Toby                   0.0               0.0               4.5              4.0                0.0               1.0
# print("\n")
# # groupby로도 가능
# print(df_movie.groupby(["critic","title"]).agg({"rating":"first"}).unstack().fillna(0))
# #                     rating
# # title         Just My Luck Lady in the Water Snakes on a Plane Superman Returns The Night Listener You Me and Dupree
# # critic
# # Claudia Puig           3.0               0.0               3.5              4.0                4.5               2.5
# # Gene Seymour           1.5               3.0               3.5              5.0                3.0               3.5
# # Jack Matthews          0.0               3.0               4.0              5.0                3.0               3.5
# # Lisa Rose              3.0               2.5               3.5              3.5                3.0               2.5
# # Mick LaSalle           2.0               3.0               4.0              3.0                3.0               2.0
# # Toby                   0.0               0.0               4.5              4.0                0.0               1.0

# print("\n=======================================================")
# print("case_ipcr_crosstab")
# print("=======================================================")
# df_ipcr = pd.read_csv("./data/ipcr.tsv", delimiter="\t")
# print(df_ipcr.head(5))
# print("\n")
# print(df_ipcr["section"].isnull().sum()) # section column의 데이터 값 중 null 개수 더하기
# print("\n")
# print(df_ipcr["ipc_class"].isnull().sum())
# print("\n")
# print(df_ipcr["subclass"].isnull().sum())
# print("\n")
# df_ipcr=df_ipcr[df_ipcr["subclass"].isnull()==False] # null인 부분 False로 바꿈
# df_ipcr["ipc_class"]=df_ipcr["ipc_class"].map(str) # 각각의 값들을 string으로 바꿈
# df_ipcr=df_ipcr[df_ipcr["ipc_class"].map(str.isdigit)] #
# df_ipcr["ipc_class"]=df_ipcr["ipc_class"].astype(int)
# two_digit_f = lambda x : '{0:02d}'.format(x) # 두개의 digit으로 바꿈
# print(two_digit_f(3))
# print("\n")
# df_ipcr["ipc_class"]=df_ipcr["ipc_class"].map(two_digit_f)
# print(df_ipcr["ipc_class"][:3])
# print("\n")
# df_ipcr["subclass"]=df_ipcr["subclass"].map(str.upper)
# # isin : 안에 저 문자들이 있는지 확인할 수 있다. 즉 숫자가 들어가있으면 제거해줌
# df_ipcr=df_ipcr[df_ipcr["subclass"].isin(list("ABCEDFGHIJKLMNOPQRSTUVWXYZ"))]
# df_ipcr["4digit"]=df_ipcr["section"]+df_ipcr["ipc_class"]+df_ipcr["subclass"]
# df_data=df_ipcr[["patent_id", "4digit"]]
# print(df_data.describe())
# print("\n")
# print(df_data[:5])
# print("\n")
# df_data=df_data.drop_duplicates(['patent_id','4digit']) # 중복제거
# print(df_data.describe())
# print("\n")
# df_data["value"]=True
# df_small_data=df_data.loc[:50000, :]
# df_matrix=pd.crosstab(df_small_data.patent_id, df_small_data["4digit"],
#         df_small_data.value, aggfunc="firsh")
# print(df_matrix.fillna(0))
# print("\n")
# print(df_matrix)
# print("\n")
# print(df_matrix.loc["3930477", :].argmax())

# print("\n=======================================================")
# print("merge_concat")
# print("=======================================================")
# # merge
# # sql에서 많이 사용하는 merge와 같은 기능
# # 두 개의 데이터를 하나로 합침
# raw_data = {
#         'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
#         'test_score': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
# df_a = pd.DataFrame(raw_data, columns=['subject_id','test_scroe'])
# print(df_a)
# print("\n")
#
# raw_data = {
#         'subject_id': ['4', '5', '6', '7', '8'],
#         'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
#         'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
# df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
# print(df_b)
# print("\n")
# # subject_id를 기준으로 (기본적으로 inner join)
# print(pd.merge(df_a, df_b, on='subject_id'))
# print("\n")
# # 두 dataframe의 column명이 다를 때 left_on과 right_on을 사용할 수 있음
# print(pd.merge(df_a, df_b, left_on='subject_id', right_on='subject_id')) # 위와 같은 결과
# print("\n")
# # left join
# print(pd.merge(df_a, df_b, on='subject_id', how='left'))
# print("\n")
# # right join
# print(pd.merge(df_a, df_b, on='subject_id', how='right'))
# print("\n")
# # outer join
# print(pd.merge(df_a, df_b, on='subject_id', how='outer'))
# print("\n")
# # inner join
# print(pd.merge(df_a, df_b, on='subject_id', how='inner'))
# print("\n")
# # 양쪽 index를 살릴 수 있려 index값을 기준으로 merge가능
# print(pd.merge(df_a, df_b, right_index=True, left_index=True))
# print("\n=======================================================\n")
#
# # concat
# raw_data = {
#         'subject_id': ['1', '2', '3', '4', '5'],
#         'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
#         'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
# df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
# print(df_a)
# print("\n")
# raw_data = {
#         'subject_id': ['4', '5', '6', '7', '8'],
#         'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
#         'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
# df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
# print(df_b)
# print("\n")
# df_new = pd.concat([df_a, df_b])
# print(df_new.reset_index())
# print("\n")
# print(df_a.append(df_b))
# print("\n")
# df_new=pd.concat([df_a, df_b], axis=1)
# print(df_new.reset_index())
# print("\n=======================================================\n")
#
# # case
# import os
# files=[file_name for file_name in os.listdir("./data") if file_name.endswith("xlsx")]
# files.remove("excel-comp-data.xlsx")
# files.remove("df_routes.xlsx")
# print(files)
# print("\n")
# df_list=[pd.read_excel("data/"+df_filename) for df_filename in files]
# status = df_list[0]
# sales = pd.concat(df_list[1:])
# print(status.head())
# print("\n")
# print(sales.head())
# print("\n")
#
# merge_df = pd.merge(status,sales, how="inner", on="account number")
# print(merge_df.head())
# print("\n")
#
# print(merge_df.groupby(["status","name_x"])["quantity", "ext price"].sum().reset_index().sort_values(
#         by=["status", "quantity"], ascending=False))

print("\n=======================================================")
print("db_persistence")
print("=======================================================")
import sqlite3
conn = sqlite3.connect("./data/flights.db")
cur = conn.cursor()
cur.execute("select * from airlines limit 5;")
result = cur.fetchall()
print(result)
print("\n")

df_airplines = pd.read_sql_query("select * from airlines;", conn)
df_airports = pd.read_sql_query("select * from airports;", conn)
print(df_airports)
print("\n")
print(df_airplines)
print("\n=======================================================\n")

# xls db_persistence
# DataFrame의 엑셀 추출 코드
# xls 엔진으로 openpyxls 또는 XlsWrite 사용
# writer = pd.ExcelWrite('./data/df_routes.xlsx', engine='xlsxwriter')
# df_routes.to_excel(writer, sheet_name='Sheet1')

# Pickle persistence
# 가장 일반적인 python 파일 persistence
# to_pickle, read_pickle 함수 사용
# df_routes.to_pickle("./data/df_routes.pickle")
# df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")
# df_routes_pickle.head()
# df_routes_pickle.describe()
