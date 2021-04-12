# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Titanic Project
# Data는 testset과 training set을 제공
# Testset으로 모델을 만든 후 trainset에 적용
# 결과제출은 [ID, 생존 예측] 형태로 제출
# 제출된 결과를 바탕으로 accuracy 점수로 등수를 산정함
# 분석가들은 기존 자신들이 시도했던 다양한 분석 방법을 사이트를 통해서 공유함.


# Load dataset
test_df = pd.read_csv("./test.csv")
train_df = pd.read_csv("./train.csv")
print(train_df.head(1))
print("\n")

print(test_df.head(1))
print("\n")

train_df.set_index('PassengerId', inplace=True)
test_df.set_index('PassengerId', inplace=True)
print(train_df.head(1))
print("\n")

print(test_df.head(1))
print("\n")

train_index = train_df.index # index값 모으기
test_index = test_df.index
y_train_df = train_df.pop("Survived") # Survived에 대한 값을 가져옴
print(y_train_df.head(3))
print("\n---------------------------------------------------\n")

# Data preprocessing
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(test_df.isnull().sum()/len(test_df)) # 널 값이 몇개 있는지 확인
print("\n")

print(train_df.isnull().sum()/len(train_df)*100) # 널 값에 대한 퍼센트
print("\n")

# Decion 1 - Drop cabin
del test_df["Cabin"]
def train_df["Cabin"] # Cabin column삭제
all_df = train_df.append(test_df)
print(all_df)
print("\n")

(all_df.isnull().sum()/len(all_df)).plot(kind='bar') # 널값이 있는지 다시 확인
plt.show()

print(len(all_df))

del all_df["Name"]
del all_df["Ticket"]
print(all_df.head())
print("\n")

all_df["Sex"]=all_df["Sex"].replace({"male":0,"femail":1}) # one_hot_encoding 시킴
print(all_df.head())
print("\n")

print(all_df["Embarked"].unique()) # Embarked : 어느 항구에서 출발했는가
print("\n")

# Embarked에 대한 one_hot_encoding한다.
all_df["Embarked"]=all_df["Embarked"].replace({"S":0,"C":1,"Q":2,np.nan:99})
print(all_df["Embarked"].unique())
print("\n")

print(all_df.head())
print("\n")

print(df.get_dummies(all_df["Embarked"], prefix="embarked"))
print("\n")

matrix_df = pd.merge(all_df, pd.get_dummies(all_df["Embarked"], prefix="embarked"),
        left_index=True, right_index=True)
print(matrix_df.head())
print("\n")

print(matrix_df.corr()) # 상관관계 보기
print("\n")

print(all_df.groupby("Pclass")["Age"].mean())
print("\n")

print(all_df.groupby("Sex")["Age"].mean())
print("\n")

print(all_df.loc[(all_df["Pclass"]==1) & (all_df["Age"].isnull()), "Age"])
print("\n")

all_df.loc[(all_df["Pclass"]==1) & (all_df["Age"].isnull()), "Age"] = 39.16
all_df.loc[(all_df["Pclass"]==2) & (all_df["Age"].isnull()), "Age"] = 29.51
all_df.loc[(all_df["Pclass"]==3) & (all_df["Age"].isnull()), "Age"] = 24.82
print(all_df.isnull().sum())
print("\n")

print(all_df.groupby("Pclass")["Fare"].mean())
print("\n")

print(all_df[all_df["Fare"].isnull()])
print("\n")

all_df.loc[all_df["Fare"].isnull(), "Fare"] = 13.30 # 평균값 넣어줌
del all_df["Embarked"] # one_hot_encoding으로 바꾸어주었기 때문에 삭제시킴
all_df["Pclass"]=all_df["Pclass"].replace({1:"A",2:"B",3:"C"})
all_df=pd.get_dummies(all_df)
print(all_df.head())
print("\n")

all_df=pd.merge(all_df, matrix_df[["embarked_0","embarked_1","embarked_2","embarked_99"]],
    left_index=True, right_index=True)
train_df = all_df[all_df.index.isin(train_index)]
test_df=all_df[all_df.index.isin(test_index)]
print(train_df.head(3))
print("\n")

print(test_df.head(3))
print("\n")

# Build Model
x_data = train_df.as_matrix()
y_data = y_train_df.as_matrix()
print(x_data.shape, y_data.shape)
print("\n")

print(y_data)
print("\n")

from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
print(cls.fit(x_data, y_data))
print("\n")

print(cls.intercept_)
print("\n")

print(cls.coef_)
print("\n")

print(cls.predict(test_df.values))
print("\n")

print(test_df.index)
print("\n")

x_test = test_df.as_matrix()
y_test = cls.predict(x_test)
print(y_test)
print("\n")

result=np.concatenate((test_index.values.reshape(-1,1),
        cls.predict(x_test).reshape(-1,1)), axis=1)
print(result[:5])
print("\n")

df_submssion = pd.DataFrame(result, columns=["PassengerId", "Survived"])
print(df_submssion)
print("\n")

print(df_submssion.to_csv("submission_result.csv", index=False))
