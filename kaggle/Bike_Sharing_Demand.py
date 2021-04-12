# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
자전거 공유 시스템은 도시 전역의 키오스크 위치 네트워크를 통해
멤버십, 대여 및 자전거 반납 절차가 자동화되는 자전거 대여 수단입니다.
이러한 시스템을 사용하여 사람들은 한 장소에서 자전거를 빌려 필요에 따라
다른 장소로 반납 할 수 있습니다. 현재 전 세계적으로 500 개 이상의 자전거 공유 프로그램이 있습니다.

이러한 시스템에서 생성 된 데이터는 여행 기간, 출발 위치, 도착 위치
및 경과 시간이 명시 적으로 기록되기 때문에 연구자들에게 매력적입니다.
따라서 자전거 공유 시스템은 도시에서 이동성을 연구하는 데 사용할 수있는
센서 네트워크로 기능합니다.
이 대회에서 참가자들은 워싱턴 D.C.의 Capital Bikeshare 프로그램에서
자전거 대여 수요를 예측하기 위해 과거 사용 패턴과 날씨 데이터를 결합해야합니다.
"""

"""
# Dataset
datetime-시간별 날짜 + 타임 스탬프
season - 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울
holiday-그날이 휴일로 간주되는지 여부
workingday-그날이 주말도 아니고 공휴일도 아닌지

weather
1 : 맑음, 약간 구름, 부분적으로 흐림, 부분적으로 흐림
2 : 안개 + 흐림, 안개 + 부서진 구름, 안개 + 약간의 구름, 안개
3 : 약한 눈, 약한 비 + 뇌우 + 흩어진 구름, 약한 비 + 흩어진 구름
4 : 폭우 + 얼음 깔판 + 뇌우 + 안개, 눈 + 안개
1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

temp - temperature in Celsius(섭씨 온도)
atemp - "feels like" temperature in Celsius (섭씨 온도 "느낌")
humidity - relative humidity (상대 습도)
windspeed - wind speed (풍속)
casual - number of non-registered user rentals initiated
(등록되지 않은 사용자 대여가 시작된 수)
registered - number of registered user rentals initiated
(시작된 등록 된 사용자 대여 수)
count - number of total rentals
(총 대여 수)
"""

"""
n is the number of hours in the test set
(n - 테스트 세트의 시간)
pi is your predicted count
(pi - 예상 개수)
ai is the actual count
(ai - 실제 개수)
log(x) is the natural logarithm
(log(x) - 자연 로그)
"""
test_df = pd.read_csv("./bike_data/test.csv", parse_dates=["datetime"])
train_df = pd.read_csv("./bike_data/train.csv", parse_dates=["datetime"])
# test데이터와 train데이터에서 0부터 시작하는 부분의 index번호가 같다
# 그래서 reset_index를 해주어 인덱스 중복을 방지
all_df = pd.concat((train_df, test_df), axis=0).reset_index()
# print(all_df.head())
# print(all_df.tail())

train_index = list(range(len(train_df))) # index값을 저장
test_index = list(range(len(train_df),len(all_df)))
# print(all_df.isnull().sum())  # test데이터에서만 없는 값들이 존재 -> null값은 존재하지 않는다.

# \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+)^2 } - RMSLE
x = np.array([np.inf, -np.inf, np.nan, -128, 128])
# print(np.nan_to_num(x))

def rmsle(y, y_):
    # nan_to_num : nan값을 매우 작은 값, inf는 매우 큰 값을 넣어줌
    log1 = np.nan_to_num(np.log(y+1))
    log2 = np.nan_to_num(np.log(y_+1))
    calc = (log1 - log2)**2
    return np.sqrt(np.mean(calc))

submission_df = pd.read_csv("./bike_data/sampleSubmission.csv")
# print(submission_df.head())

rmsle(submission_df["count"].values,
       np.random.randint(0,100, size=len(submission_df))))

del all_df["casual"]
del all_df["registered"]
del all_df["index"]

# one hot encoding - 만들어줬다고 원래 있던 값(season이나 weather)를 삭제하지말자.
pre_df = all_df.merge(pd.get_dummies(all_df["season"], prefix="season"),
                      left_index=True, right_index=True)
# print(pre_df1.head())

pre_df = pre_df.merge(pd.get_dummies(all_df["weather"], prefix="weather"),
                      left_index=True, right_index=True)
# print(pre_df.head())



# print(pre_df["datetime"].unique()) # len을 하면 전체 index값과 같음 그래서 datetime 자체가 유니크함
# ['2011-01-01T00:00:00.000000000' '2011-01-01T01:00:00.000000000'
#  '2011-01-01T02:00:00.000000000' ... '2012-12-31T21:00:00.000000000'
#  '2012-12-31T22:00:00.000000000' '2012-12-31T23:00:00.000000000']
pre_df["year"] = pre_df["datetime"].dt.year
pre_df["month"] = pre_df["datetime"].dt.month
pre_df["day"] = pre_df["datetime"].dt.day
pre_df["hour"] = pre_df["datetime"].dt.hour
pre_df["weekday"] = pre_df["datetime"].dt.weekday # weekday = dayofweek (버전차이)

pre_df=pre_df.merge(pd.get_dummies(pre_df["weekday"], prefix="weekday"),
                    left_index=True, right_index=True)
# print(pre_df.head())

# print(pre_df.dtypes)

category_variable_list = ["season","weather","workingday","season_1",
                          "season_2","season_3","season_4","weather_1",
                          "weather_2","weather_3","weather_4","year",
                          "month","day","hour","weekday","weekday_0",
                          "weekday_1","weekday_2","weekday_3","weekday_4",
                          "weekday_5","weekday_6"]
for var_name in category_variable_list:
    pre_df[var_name] = pre_df[var_name].astype("category") # 데이터타입 변경

# print(pre_df.dtypes)

train_df = pre_df.iloc[train_index]

# fig, axes = plt.subplots(nrows=3,ncols=3)
# fig.set_size_inches(12, 5)
# axes = plt.subplots(nrows=3, ncols=3)
# axes[0][0].bar(train_df["year"], train_df["count"])
# axes[0][1].bar(train_df["weather"], train_df["count"])
# axes[0][2].bar(train_df["workingday"], train_df["count"])
# axes[1][0].bar(train_df["holiday"], train_df["count"])
# axes[1][1].bar(train_df["weekday"], train_df["count"])
# axes[1][2].bar(train_df["month"], train_df["count"])
# axes[2][0].bar(train_df["day"], train_df["count"])
# axes[2][1].bar(train_df["hour"], train_df["count"])
# plt.show()

# 월별로 그룹하면서 count로 구별
serires_data = train_df.groupby(["month"])["count"].mean()
# print(serires_data.index.tolist()[:5])

# fig, ax = plt.subplots()
# ax.bar(range(len(serires_data)), serires_data)
# fig.set_size_inches(12, 5)
# plt.show()

# import seaborn as sn
# fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
# fig.set_size_inches(12, 5)
# sn.regplot(x="temp", y="count", data=train_df, ax=ax1)
# sn.regplot(x="windspeed", y="count", data=train_df, ax=ax2)
# sn.regplot(x="humidity", y="count", data=train_df, ax=ax3)
# plt.show()

# print(category_variable_list)

# corrMatt = train_df[["temp", "atemp", "humidity", "windspeed", "count"]].corr()
# mask = np.array(corrMatt)
# make[np.tril_indices_from(mask)] = False
# fig,ax = plt.subplots()
# fig.set_size_inches(20, 10)
# sn.heatmap(corrMatt, maks=mask, vmax=.8, square=True, annot=True)
# plt.show()

# print(category_variable_list[:5])

continuous_variable_list = ["temp", "humidity", "windspeed", "atemp"]
season_list = ['season_1', 'season_2', 'season_3', 'season_4']
weather_list = ['weather_1', 'weather_2', 'weather_3', 'weather_4']
weekday_list = ['weekday_0','weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6']
category_varialbe_list = ["season","holiday","workingday","weather","weekday","month","year","hour"]

all_variable_list = continuous_variable_list + category_variable_list
all_variable_list.append(season_list)
all_variable_list.append(weather_list)
all_variable_list.append(weekday_list)

# print(all_variable_list)

number_of_variable = len(all_variable_list)
# print(number_of_variable)

variable_combinations = []
import itertools
for L in range(8, number_of_variable+1):
    # itertools.combinations : combination의 개수만큼 모든 경우의 수를 다 뽑을 수 있음
    # all_variable_list : 15, L : 우리가 입력한 값
    # 15C8 , 15C9 ...
    for subset in itertools.combinations(all_variable_list, L):
        temp=[]
        for variable in subset:
            if isinstance(variable, list):
                for value in variable:
                    temp.append(value)
            else:
                temp.append(variable)
        variable_combinations.append(temp)

# print(len(variable_combinations))

del pre_df["count"]



from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFlod
import datetime

kf = KFlod(n_splits=10)

y = train_df["count"].values
final_output = []
models = []

# RMSLE -> 기준이 작을 수록 좋음
# Ridge, Lasso, LR를 사용
# variable_combination : 10~15 생성

print(len(variable_combinations))
ts = datetime.datetime.now()
for i, combination in enumerate(variable_combinations):
    lr = LinearRegression(n_jobs=8)
    ridge = Ridge()
    lasso = Lasso()

    lr_result = []
    ridge_result = []
    lasso_result = []

    target_df = pre_df[combination]
    ALL = target_df.values
    std = StandardScaler()
    std.fit(ALL)
    ALL_scaled = std.transform(ALL)
    X = ALL_scaled[train_index]

    for train_data_index, test_data_index in kf.split(X):
        X_train = X[train_data_index]
        X_test = X[test_data_index]
        y_train = y[train_data_index]
        y_test = y[test_data_index]

        lr.fit(X_train, y_train)
        result = rmsle(y_test, le.predict(X_test))
        lr_result.append(result)

        ridge.fit(X_train, y_train)
        result = rmsle(y_test, ridge.predict(X_test))
        ridge_result.append(result)

        lasso.fit(X_train, y_train)
        result = rmsle(y_test, lasso.predict(X_test))
        lasso_result.append(result)

    final_output.append([i, np.mean(lr_result), np.mean(ridge_result), np.mean(lasso_result)])
    models.append([lr,ridge,lasso])
    if i % 100 == 0: # 기억하자!
        tf = datetime.datetime.now() # 현재 시간
        te = tf - ts
        print(i, te)
        ts = datetime.datetime.now() # 100개할때마다 총 걸린시간 찍음

labels = ["combination", "lr", "ridge", "lasso"]
from pandas import DataFrame
result_df = DataFrame(final_output, columns=labels) # 모델 생성
print(result_df.head())

print("min: ", result_df.min())
print(result_df["lasso"].sort_values().head())
print(variable_combinations[4752])

target_df = pre_df[variable_combinations[4752]]
ALL = target_df.values
std = StandardScaler()
std.fit(ALL)
ALL_scaled = std.transform(ALL)
# test data 만듬
X_submission_test = ALL_scaled[test_index]
X_submission_test.shape

print(X_submission_test)
# 제일 작은 값(위에서 min으로 구한 것)
# ex) lasso 중 8번째 -> models[8][2]
print(models[4752][2])

final_result = models[4752][2].predict(X_submission_test)
final_result[final_result < 0] = 0

print(final_result)
print(pre_df.iloc[test_index]["datetime"].head())
data = {"datetime":pre_df.iloc[test_index]["datetime"], "count":final_result}
df_submission = DataFrame(data, columns=["datatime", "count"])
print(df_submission.head())
df_submission.set_index("datetime").to_csv("submission_lasso_data.csv")
