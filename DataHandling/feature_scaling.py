#--coding:utf-8--
import pandas as pd
import numpy as np

# 두 변수 중 하나의 값의 크기가 너무 클 때,
# 즉, 몸무게와 키가 변수일 때 키가 영행을 많이 준다.

# Feature scaling
# Feature간의 최대-최소값의 차이를 맞춘다.

# Feature scaling 전략
# Min-Max Normalization
# 기존 변수에 범위를 새로운 최대-최소로 변경
# 일반적으로 0과 1 사이 값으로 변경함

# Standardization (Z-score Normalization)
# 기존 변수의 범위에 정규 분포로 변환
# 실제 min-max의 값을 모를 때 활용 가능

df = pd.DataFrame({'A':[14.00,90.20,90.95,96.27,91.21], 'B':[103.02,107.26,110.35,114.23,114.68],
    'C':['big','small','big','small','small']})
print(df)
print("\n")

print(dp["A"])
print("\n")

print(df["A"]-df["A"].min())
print("\n")

print((df["A"]-df["A"].min())/(df["A"].max()-df["A"].min()))
print("\n")

df["A"]=(df["A"]-df["A"].min())/(df["A"].max()-df["A"].min())*(5-1)+1
print(df)
print("\n")

print(df["B"].mean(), df["B"].std())
print("\n")

df["B"]=(df["B"]-df["B"].mean())/(df["B"].std())
print(df)
print("\n")

def feture_scaling(df, scaling_strategy="min-max", column=None):
    if column == None:
        column = [column_name for column_name in df.columns]
    for column_name in column:
        if scaling_strategy == "min-max":
            df[column_name] = (df[column_name]-df[column_name].min())/
                    (df[column_name].max() - df[column_name].min())
        elif scaling_strategy == "z-score":
            df[column_name] = (df[column_name]-df[column_name].mean())/(df[column_name].std())
    return df
df = pd.DataFrame({'A':[14.00,90.20,90.95,96.27,91.21],'B':[103.02,107.26,110.35,114.23,114.68], 'C':['big','small','big','small','small']})
print(feture_scaling(df,column=["A","B"]))
print("\n")

df=pd.io.parsers.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
            header=None, usecols=[0,1,2])
df.columns=['Class label','Alcohol','Malic acid']
print(df.head())
print("\n")

df=feature_scaling(df,"min-max",column=['Alcohol','Malic acid'])
print(df.head())
print("\n")

from sklearn import preprocessing
# Feature scaling with sklearn
# Label encoder와 마찬가지로, sklearn도 feature scale 지원
# MinMaxScaler와 StandardScaler 사용
# Preprocession은 모두 fit -> transform의 과정을 거침
# 이유는 label encoder와 동일
# 단, scaler는 한번에 여러 column을 처리 가능
df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )
df.columns=['Class label', 'Alcohol', 'Malic acid']
std_scaler=preprocessing.StandardScaler().fit(df[['Alcohol','Malic acid']])
df_std=std_scaler.transform(df[['Alcohol','Malic acid']])
print(df_std)
print("\n")

minmax_scaler=preprocessing.MinMaxScaler().fit(df[['Alcohol','Malic acid']])
print(minmax_scaler.transform(df[['Alcohol','Malic acid']]))
print("\n")

print(df_minmax)
