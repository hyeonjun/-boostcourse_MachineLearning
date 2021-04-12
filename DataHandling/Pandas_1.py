# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
# print("=======================================================")
# print("data_loading")
# print("=======================================================")
# data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
# # csv 타입 데이터 로드, separate는 빈칸으로 다 가져오기위해사용, Column은 없음
# df_data = pd.read_csv(data_url, sep='\s+', header=None)
# print(df_data.head()) # head는 처음 다섯줄을 찍어라
# # Column이 0,1,2,~~였던 것들을 이름 지정
# df_data.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO' ,'B', 'LSTAT', 'MEDV']
# print(df_data.head())
# print(type(df_data.values)) # numpy.ndarray 즉, numpy의 array타입이라는 소리

print("\n=======================================================")
print("pandas_Series")
print("=======================================================")
list_data = [1,2,3,4,5]
example_obj = Series(data=list_data) # list_data도 가능하고, dict도 가능
print(example_obj)
# index data
# 0      1
# 1      2
# 2      3
# 3      4
# 4      5
# dtype: int64 (data type)
print("\n=======================================================\n")

list_name=["a","b","c","d","e"]
example_obj=Series(data=list_data, dtype=np.float32, index=list_name) # index명 지정
print(example_obj)
print("\n")
print(example_obj["a"]) # data index에 접근
print("\n")
example_obj["a"] = 3.2
print(example_obj)
print("\n")
print(example_obj[example_obj > 2])
print("\n")
print(example_obj*2)
print("\n")
print(np.exp(example_obj))
print("\n")
print("b" in example_obj) # True
print("\n")
print(example_obj.to_dict()) # {'a': 3.200000047683716, 'c': 3.0, 'b': 2.0, 'e': 5.0, 'd': 4.0}
print("\n")
print(example_obj.values) # [3.2 2. 3. 4. 5. ]
print("\n")
print(example_obj.index) # Index([u'a', u'b', u'c', u'd', u'e'], dtype='object')
print("\n")
example_obj.name = "number"
example_obj.index.name = "alpabet"
print(example_obj)
# alpabet
# a    3.2
# b    2.0
# c    3.0
# d    4.0
# e    5.0
# Name: number, dtype: float32
print("\n=======================================================\n")

dict_data_1 = {"a":1, "b":2, "c":3, "d":4, "e":5}
indexes=["a","b","c","d","e","f","g","h"]
series_obj_1 = Series(dict_data_1, index=indexes)
print(series_obj_1)
# a    1.0
# b    2.0
# c    3.0
# d    4.0
# e    5.0
# f    NaN
# g    NaN
# h    NaN
# dtype: float64
print("\n=======================================================")
print("pandas_dataframe")
print("=======================================================")
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
df = pd.DataFrame(raw_data, columns=['first_name', 'last_name', 'age', 'city'])
print(df)
print("\n")
print(DataFrame(raw_data, columns= ["age","city"]))
print("\n")
df = DataFrame(raw_data, columns= ['first_name', 'last_name', 'age', 'city','debt'])
print(df)
print("\n")
print(df.first_name)
print("\n")
print(df["first_name"]) # print(df.first_name)와 같은 결과
# 0    Jason
# 1    Molly
# 2     Tina
# 3     Jake
# 4      Amy
# Name: first_name, dtype: object
print("\n")
print(df.loc[1]) # loc : index의 location 가지고 특정 위치의 데이터에 접근
print("\n")
print(df["age"].iloc[1:]) # iloc: index의 position
print("\n")
s = pd.Series(np.nan, index=[49,48,47,46,45,1,2,3,4,5]) # np.nan은 값을 NaN 으로
print(s)
print("\n")
print(s.iloc[3:]) # index의 위치가 3부터 출력
print("\n")
print(s.loc[:3]) # index가 3인 곳까지만 출력
print("\n")
print(df.age>40)
print("\n")
df.debt = df.age>40 # colum에 새로운 데이터 할당
print(df)
#   first_name last_name  age           city   debt
# 0      Jason    Miller   42  San Francisco   True
# 1      Molly  Jacobson   52      Baltimore   True
# 2       Tina       Ali   36          Miami  False
# 3       Jake    Milner   24        Douglas  False
# 4        Amy     Cooze   73         Boston   True
print("\n=======================================================\n")

values = Series(data=["M", "F", "F"], index=[0,1,3])
print(values)
print("\n")
df["sex"] = values # sex라는 column만들고 values에 할당된 index의 위치에 값들을 삽입
print(df)
print("\n")
print(df.T) # index와 column 위치 변경
print("\n")
print(df.values) # 데이터만 array형으로 뽑기
print("\n")
print(df.to_csv()) # csv형으로 column과 데이터 모두 가져옴
# ,first_name,last_name,age,city,debt,sex
# 0,Jason,Miller,42,San Francisco,True,M
# 1,Molly,Jacobson,52,Baltimore,True,F
# 2,Tina,Ali,36,Miami,False,
# 3,Jake,Milner,24,Douglas,False,F
# 4,Amy,Cooze,73,Boston,True,
print("\n")
del df["debt"] # debt라는 column 삭제
print(df)
print("\n")
pop = {'Nevada': {2001: 2.4, 2002: 2.9}, # dict타입 DataFrame
 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
print(DataFrame(pop))
#       Nevada  Ohio
# 2000     NaN   1.5
# 2001     2.4   1.7
# 2002     2.9   3.6

print("\n=======================================================")
print("data_selection")
print("=======================================================")
df = pd.read_excel("./data/excel-comp-data.xlsx")
print(df.head(3)) # 한개의 coulmn 선택 시
print("\n")
print(df[["account", "street", "state"]].head(3)) # 1개 이상의 column선택
print("\n")
print(df[:10]) # column 이름없이 사용하는 index number는 row 기준이다.
# 즉, 0부터 9번째까지의 데이터 출력
print("\n")
print(df["account"][:3]) # account라는 column을 0부터 2번째까지의 데이터 출력
print("\n")
account_serires = df["account"]
print(account_serires[[0,1,2]]) # index의 번호를 넣어 해당 index의 값들을 가져올 수 있음
print("\n")
print(account_serires[account_serires<250000])
print("\n")
df.index =df["account"] # index를 account로 함
print(df.head())
del df["account"] # account 삭제(현재 두 개이기 때문)
print(df.head())
print("\n")
print([["name", "street"]][:2]) # name과 street이라는 column명을 가진 것들을 0에서 1번째까지 출력
print("\n")
print(df.loc[[211829,320563],["name","street"]]) # index name과 Column명
print("\n")
print(df.iloc[:2,:1]) # index number와 column number
print("\n")
print(df[["name", "street"]].iloc[:10]) # column name, index number
print("\n")
df.index=list(range(0,15)) # reindex
print(df.head())
print("\n")
print(df.drop(1)) # index number로 drop
#                                 name                                street               city         state  postal-code     Jan     Feb     Mar
# 0         Kerluke, Koepp and Hilpert                    34456 Sean Highway         New Jaycob         Texas        28752   10000   62000   35000
# 2         Bashirian, Kunde and Price  62184 Schamberger Underpass Apt. 231     New Lilianland          Iowa        76517   91000  120000   35000
# 3        D'Amore, Gleichner and Bode           155 Fadel Crescent Apt. 144         Hyattburgh         Maine        46021   45000  120000   10000
print("\n")
print(df.drop([0,1, 2,3])) # 여러 개도 가능
print("\n")
print(df.drop("city", axis=1)) # axis 지정으로 축을 기준으로 drop, column중에 "city"
#           "city", "state" 도 가능
# inplace=True를 넣어주면 실제 데이터를 삭제시킴
print("\n=======================================================\n")

matrix = df.as_matrix()
print(matrix[:3])
# [[u'Kerluke, Koepp and Hilpert' u'34456 Sean Highway' u'New Jaycob'
#   u'Texas' 28752L 10000L 62000L 35000L]
#  [u'Walter-Trantow' u'1311 Alvis Tunnel' u'Port Khadijah'
#   u'NorthCarolina' 38365L 95000L 45000L 35000L]
#  [u'Bashirian, Kunde and Price' u'62184 Schamberger Underpass Apt. 231'
#   u'New Lilianland' u'Iowa' 76517L 91000L 120000L 35000L]]
print("\n")
print(matrix[:,-3:])
print("\n")
print(matrix[:,-3:].sum(axis=1))

print("\n=======================================================")
print("Sries_operation")
print("=======================================================")
s1 = Series(range(1,6), index=list("abced"))
print(s1)
print("\n")
s2 = Series(range(5,11), index=list("bcedef"))
print(s2)
print("\n")
print(s1.add(s2))
print("\n")
print(s1+s2) # 위와 같은 결과
# index 기준으로 연산 수행
# 겹치는 index가 없을경우 NaN으로 반환

print("\n=======================================================")
print("Dataframe_operation")
print("=======================================================")
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list("abc"))
print(df1)
print("\n")
df2 = DataFrame(np.arange(16).reshape(4,4), columns=list("abcd"))
print(df2)
print("\n")
print(df1+df2)
print("\n")
print(df1.add(df2,fill_value=0))
# df는 coulmn과 index를 모두 고려해야함
# add operation을 쓰면 NaN값 0으로 변환(fill_value)
# Operation Types : add, sub, div, mul

print("\n=======================================================")
print("Series_DataFrame_operation")
print("=======================================================")
df = DataFrame(np.arange(16).reshape(4,4), columns=list("abcd"))
s = Series(np.arange(10,14), index=list("abcd"))
print(df+s)
# column을 기준으로 broadcasting이 발생

print("\n=======================================================")
print("map_apply_lambda")
print("=======================================================")
# map for series
# Pandas의 series type의 데이터에도 map 함수 사용가능
# function 대신 dict, sequence형 자료등으로 대체 가능
s1 = Series(np.arange(10)) # 0~9
print(s1.head(5))
print("\n")
print(s1.map(lambda x: x**2).head(5))
print("\n")
z = {1: 'A', 2: 'B', 3: 'C'}
print(s1.map(z).head(5)) # 없는 값은 NaN
print("\n")
s2 = Series(np.arange(10,20)) # 10~19
print(s1.map(s2)) # 같은 index 번호끼리
print("\n=======================================================\n")

df = pd.read_csv("./data/wages.csv")
print(df.head(5))
print("\n")
print(df.sex.unique()) # unique(): series data의 유일한 값을 list로 반환 ['male' 'female']
print("\n")
print(df.sex.replace({"male":0, "female":1}).head(5))
print("\n")
df.sex.replace(["male", "female"],[0,1], inplace=True) # inplate로 실제 df에 데이터 적용
print(df.head(5))
print("\n=======================================================\n")

# apply for DataFrame
# map과 달리, Series 전체(column)에 해당 함수를 적용
# 입력값이 series 데이터로 입력받아 handling 가능
df = pd.read_csv("./data/wages.csv")
print(df.head())
print("\n")
df_info=df[["earn","height","age"]]
df_info.head()
f = lambda x: x.max()-x.min() # 각 컬럼별로 max값 - min값
print(df_info.apply(f))
print("\n")
print(df_info.apply(sum))
print("\n")
print(df_info.sum()) # 위와 같은 결과 출력
print("\n")
# scalar 값 이외에 series값의 반환도 가능
def f(x):
    return Series([x.min(), x.max(), x.mean()], index=["min","max","mean"])
print(df_info.apply(f))
print("\n=======================================================\n")

# applymap for dataframe
# series 단위가 아닌 element 단위로 함수를 적용함
# series 단위에 apply를 적용시킬때와 같은 효과
f = lambda x: -x
print(df_info.applymap(f).head(5))
print("\n")
print(df_info["earn"].apply(f).head(5))
print("\n=======================================================\n")

# describe
# Numeric type 데이터의 요약 정보를 보여줌
df = pd.read_csv("./data/wages.csv")
print(df.head())
#            earn  height     sex   race  ed  age
# 0  79571.299011   73.89    male  white  16   49
# 1  96396.988643   66.23  female  white  16   62
# 2  48710.666947   63.77  female  white  16   33
# 3  80478.096153   63.22  female  other  16   95
# 4  82089.345498   63.08  female  white  17   43
print("\n")
print(df.describe()) # 각 컬럼별로 통계자료를 뽑아줌
#                 earn       height           ed          age
# count    1379.000000  1379.000000  1379.000000  1379.000000
# mean    32446.292622    66.592640    13.354605    45.328499
# std     31257.070006     3.818108     2.438741    15.789715
# min       -98.580489    57.340000     3.000000    22.000000
# 25%     10538.790721    63.720000    12.000000    33.000000
# 50%     26877.870178    66.050000    13.000000    42.000000
# 75%     44506.215336    69.315000    15.000000    55.000000
# max    317949.127955    77.210000    18.000000    95.000000
print("\n=======================================================\n")

# unique - serise data의 유일한 값을 list로 반환
print(df.race.unique()) # ['white' 'other' 'hispanic' 'black']
print("\n")
print(np.array(dict(enumerate(df["race"].unique())))) # dict type으로 index
# {0: 'white', 1: 'other', 2: 'hispanic', 3: 'black'}
print("\n")
value = list(map(int, np.array(list(enumerate(df["race"].unique())))[:,0].tolist()))
key = np.array(list(enumerate(df["race"].unique())), dtype=str)[:,1].tolist()
print(value,key) # label index값과 label 값 각각 추출
# ([0, 1, 2, 3], ['white', 'other', 'hispanic', 'black'])
