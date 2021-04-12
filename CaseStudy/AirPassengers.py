# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

# Data
# 월별, 년도별 승객은 얼마나 될까?
# 이전달과 이번달의 승객 차이는? 상승율은?
# 누적 승객수, 최고, 최소 승객수는?


df_time_series=pd.read_csv("./AirPassengers.csv")
# print(df_time_series.head())

# print("\n")

df_time_series["step"]=range(len(df_time_series)) # step column을 만들어서 열 값을 넣음
# print(df_time_series.head())
#      Month  #Passengers  step
# 0  1949-01          112     0
# 1  1949-02          118     1
# 2  1949-03          132     2
# 3  1949-04          129     3
# 4  1949-05          121     4

# print("\n")

# cum_sum이라는 column에 #Passengers에 있는 데이터값들을 index가 증가하면서 합침
df_time_series["cum_sum"]=df_time_series["#Passengers"].cumsum()
# cum_max라는 column에 #Passengers에서 현재 읽은 데이터들 중 가장 큰 값을 넣음
df_time_series["cum_max"]=df_time_series["#Passengers"].cummax()
# cum_min -> cum_max와 반대
df_time_series["cum_min"]=df_time_series["#Passengers"].cummin()
# print(df_time_series.head())
#      Month  #Passengers  step  cum_sum  cum_max  cum_min
# 0  1949-01          112     0      112      112      112
# 1  1949-02          118     1      230      118      112
# 2  1949-03          132     2      362      132      112
# 3  1949-04          129     3      491      132      112
# 4  1949-05          121     4      612      132      112
# print("\n")

# split으로 Month에 해당되는 데이터들을 - 를 기준으로 잘라서 temp_date에 넣음
# temp_date에 있는 값들을 list형태로 바꾸고 array형식으로 만듬
temp_date = df_time_series["Month"].map(lambda x : x.split('-'))
temp_date = np.array(temp_date.values.tolist())
# print(temp_date[:5])
# [['1949' '01']
#  ['1949' '02']
#  ['1949' '03']
#  ['1949' '04']
#  ['1949' '05']]
# print("\n")

# df_time_series에 year와 month라는 column을 만들고
# temp_date의 모든 데이터들(:) 중 index [0]번째인 년도는 year에
# index [1]번째인 월은 month에 삽입
df_time_series["year"] = temp_date[:, 0]
df_time_series["month"] = temp_date[:, 1]
# print(df_time_series.head())
#      Month  #Passengers  step  cum_sum  cum_max  cum_min  year month
# 0  1949-01          112     0      112      112      112  1949    01
# 1  1949-02          118     1      230      118      112  1949    02
# 2  1949-03          132     2      362      132      112  1949    03
# 3  1949-04          129     3      491      132      112  1949    04
# 4  1949-05          121     4      612      132      112  1949    05
# print("\n")

# diff라는 column 생성 후 #Passengers에서 자신의 데이터와 이전 인덱스에 해당하는 데이터와의
# 차이를 diff에 삽입
df_time_series["diff"] = df_time_series["#Passengers"].diff().fillna(0)
# print(df_time_series.head())
#      Month  #Passengers  step  cum_sum  cum_max  cum_min  year month  diff
# 0  1949-01          112     0      112      112      112  1949    01   0.0
# 1  1949-02          118     1      230      118      112  1949    02   6.0
# 2  1949-03          132     2      362      132      112  1949    03  14.0
# 3  1949-04          129     3      491      132      112  1949    04  -3.0
# 4  1949-05          121     4      612      132      112  1949    05  -8.0
# print("\n")


# pct_chang() : (현재값-이전값)/이전값
# map(lambda x : x*100).map(lambda x : " %.2f" % x)를 사용하여
# 100을 곱하고(퍼센트구할때 *100하는 것임) 소수점 둘째자리까지만 표시하도록 한다.
df_time_series["pct"]=df_time_series["#Passengers"].pct_change().map(lambda x :
            x*100).map(lambda x : "%.2f" % x)
print(df_time_series.head())
#      Month  #Passengers  step  cum_sum  cum_max  cum_min  year month  diff    pct
# 0  1949-01          112     0      112      112      112  1949    01   0.0    nan
# 1  1949-02          118     1      230      118      112  1949    02   6.0   5.36
# 2  1949-03          132     2      362      132      112  1949    03  14.0  11.86
# 3  1949-04          129     3      491      132      112  1949    04  -3.0  -2.27
# 4  1949-05          121     4      612      132      112  1949    05  -8.0  -6.20





















print("\n")
