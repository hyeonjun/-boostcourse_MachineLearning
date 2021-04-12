#--coding:utf-8--
import pandas as pd
import numpy as np

# Data quality problems
# 데이터의 최대/최소가 다름 -> Scale에 따른 y값에 영향
# Ordinary 또는 Nominal한 값들의 표현은 어떻게?
# 잘 못 기입된 값들에 대한 처리
# 값이 없을 경우는 어떻게 해야하는가?
# 극단적으로 큰 값 또는 작은 값들은 그대로 놔둬야 하는가?

# Data preprocessing issue
# 데이터가 빠진 경우(결측치의 처리)
# 라벨링된 데이터(category) 데이터의 처리
# 데이터의 scale의 차이가 매우 크게 날 경우

# 데이터가 없을 때 할 수 있는 전략
# 데이터가 없으면 sample을 drop
# 데이터가 없는 최소 개수를 정해서 sample을 drop
# 데이터가 거의 없는 feature는 feature 자체를 drop
# 최빈값, 평균값으로 비어있는 데이터를 채우기

# print("\n----------------------------------------\n")
# print("missing_value")
# print("\n----------------------------------------\n")
# raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],
#         'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
#         'age': [42, np.nan, 36, 24, 73],
#         'sex': ['m', np.nan, 'f', 'm', 'f'],
#         'preTestScore': [4, np.nan, np.nan, 2, 3],
#         'postTestScore': [25, np.nan, np.nan, 62, 70]}
# df = pd.DataFrame(raw_data, columns=['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
# print(df)
# print("\n")
# # print("\n-----------------------------------------------\n")
#
# print(df.isnull().sum())
# print("\n")
#
# df_no_missing = df.dropna() # null값을가진 index행은 삭제
# print(df_no_missing)
# print("\n")
#
# # 모두 null값을 가진 index행 삭제
# df_cleaned = df.dropna(how='all')
# print(df_cleaned)
# print("\n")
#
# df['location'] = np.nan # location이라는 column 추가하고 값은 모두 null값
# print(df)
# print("\n")
#                 # axis = 1 이면 column
# print(df.dropna(axis=1, how='all')) # 모두 null값을 가진 column 제외
# print("\n")
#         # axis=0이면 index(row), thresh=1이라면 데이터가 최소 1개이상 있는지
# print(df.dropna(axis=0, thresh=1))
# print("\n")
#
# print(df.dropna(thresh=5))
# print("\n")
#
# print(df.fillna(0)) # null값인 애들 0으로 표시
# print("\n")
#
# print(df["preTestScore"].mean()) # mean : 평균값
# print("\n")
#
# print(df["postTestScore"].median()) # median : 중위값(일렬로 데이터를 나열했을 때의 중간에 위치한 값)
# print("\n")
#
# print(df["postTestScore"].mode()) # mode : 최빈값(가장 자주 나온 값)
# print("\n")
#
# print(df["preTestScore"])
# print("\n")
#
# print(df["preTestScore"].fillna(df["preTestScore"].mean(), inplace=True))
# print("\n")
# print(df)
# print("\n")
#
# print(df.groupby("sex")["postTestScore"].transform("mean"))
# print("\n")
#
# print(df["postTestScore"].fillna(df.groupby("sex")["postTestScore"].transform("mean"), inplace=True))
# print("\n")
# print(df)
# print("\n")
#
# print(df[df['age'].notnull() & df['sex'].notnull()])


# print("\n----------------------------------------\n")
# print("categorical_data")
# print("\n----------------------------------------\n")

# 이산형 데이터를 어떻게 처리할까? -> One-Hot Encoding
# {Green, Blue, Yellow}
# {Green} -> {1,0,0} / {Blue} -> ....
# 실제 데이터 set의 크기만큼 Binary Feature를 생성

# edges = pd.DataFrame({'source': [0, 1, 2],
#                    'target': [2, 2, 3],
#                        'weight': [3, 4, 5],
#                        'color': ['red', 'blue', 'blue']})
# print(edges)
# print("\n")
#
# print(edges.dtypes)
# print("\n")
#
# print(edges["color"])
# print("\n")
#
# print(pd.get_dummies(edges)) # get_dummies로 one-hot encoding 시켜버림
# #    source  target  weight  color_blue  color_red
# # 0       0       2       3           0          1
# # 1       1       2       4           1          0
# # 2       2       3       5           1          0
# print("\n")
#
# print(pd.get_dummies(edges["color"])) # 이렇게 하고 merge 시키면 됨
# print("\n")
#
# print(pd.get_dummies((edges[["color"]])))
# print("\n")
#
# weight_dict = {3:"M", 4:"L", 5:"XL"}
# edges["weight_sign"] = edges["weight"].map(weight_dict)
# print(edges)
# print("\n")

#
# weight_sign = pd.get_dummies(edges["weight_sign"])
# print(weight_sign)
# print("\n")
#
# print(pd.concat([edges, weight_sign], axis=1))
# print("\n")
#
# print(pd.get_dummies(edges).values)
# print("\n----------------------------------------\n")
#
# raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
#         'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
#         'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
#         'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
#         'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
# df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
# print(df)
# print("\n")
#
# bins = [0,25,50,75,100]
# group_name = ['Low','Okay', 'Good','Great']
# categories = pd.cut(df['postTestScore'], bins, labels=group_name)
# print(categories)
# print("\n")
#
# df['categories'] = pd.cut(df['postTestScore'], bins, labels=group_name)
# print(pd.value_counts(df['categories']))
# print("\n")
#
# print(pd.get_dummies(df))
# print("\n----------------------------------------\n")
#
# # using scikit-learn preprocessing
# Scikit-learn의 preprocessing 패키지도 label, one-hot 지원
# raw_example = df.as_matrix()
# print(raw_example[:3])
# print("\n")
#
# data = raw_example.copy()
#
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder() # encoder 생성
# print(raw_example[:,0])
# print("\n")
#
# print(le.fit(raw_example[:,0])) # Data에 맞게 encoding fitting
# print("\n")
#
# print(le.classes_)
# print("\n")
#
# print(le.transform(raw_example[:,0])) # 실제 데이터를 labelling data로 변환
# print("\n")
# Label encoder의 fit과 transform의 과정이 나눠진 이유는
# 새로운 데이터 입력 시, 기존 lablelling 규칙을 그대로 적용할 필요가 있음
# Fit은 규칙을 생성화는 과정이고
# Transform은 규칙을 적용하는 과정이다
# Fit을 통해 규칙이 생성된 labelencoder는 따로 저장하여
# 새로운 데이터를 입력할 경우 사용할 수 있다
# Encoder들을 실제 시스템에 사용할 경우 pickle화 필요하다.

#
# data[:,0]=le.transform(raw_example[:,0])
# print(data[:3])
# print("\n")
#
# label_column = [0,1,2,5]
# label_encoder_list = []
# for column_index in label_column:
#     le = preprocessing.LabelEncoder()
#     le.fit(raw_example[:, column_index])
#     data[:, column_index]=le.transform(raw_example[:,column_index])
#     label_encoder_list.append(le)
#     del le
# print(data[:3])
# print("\n")
#
# print(label_encoder_list[0].transform(raw_example[:10,0]))
# print("\n")
#
# one_hot_enc=preprocessing.OneHotEncoder()
# print(data[:,0].reshape(-1,1))
# print("\n")
#
# print(one_hot_enc.fit(data[:,0].reshape(-1,1)))
# print("\n")
#
# print(one_hot_enc.n_values_) # 몇개의 값이 있는지
# print("\n")
#
# print(one_hot_enc.active_features_)  # 몇 개의 0,1,2라는 feature들이 있는가
# print("\n")
#
# print(data[:,0].reshape(-1,1))
# print("\n")
#
# onehotlabels=one_hot_enc.transform(data[:,0].reshape(-1,1)).toarray()
# print(onehotlabels)
