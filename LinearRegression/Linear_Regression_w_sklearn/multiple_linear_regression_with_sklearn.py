# -*- coding:utf-8 -*-
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston() # boston은 dict타입으로 되어있음
# print(boston.keys())
### dict_keys(['data', 'target', 'feature_names', 'DESCR'])
# print(boston["data"])


x_data = boston.data
y_data = boston.target.reshape(boston.target.size, 1)
print(y_data.shape)

from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,5)).fit(x_data)
# standard_scale = preprocessing.StandardScaler().fit(x_data)
x_scaled_data = minmax_scale.transform(x_data)

print(x_scaled_data[:3])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled_data, y_data, test_size=0.33)

# print(len(x_scaled_data))
### 506
# print(len(X_train))
### 399
# print(len(X_test))
## 167
print(X_train, X_test, y_train, y_test)

from sklearn import linear_model
regr = linear_model.LinearRegression(fit_intercept=True,
                                     normalize=False,
                                     copy_X=True,
                                     n_jobs=8)
                                     # n_jobs -> number of jobs : 프로그램을 돌릴때 cpu 개수를 몇개 쓸건지 정함
regr.fit(X_train, y_train) # fitting 시켜서 모델을 생성한다.
print(regr) # 모델을 생성했다는 것은 weight의 값들을 다 정해줬다는 의미.

# the coefficients
# ^y = w0 + w1x1 + w2x2 + ... + w13x13
# regr.intercept_ : w0에 해당하는 값
# regr.coef_ : w1~w13에 해당하는 값
print('Coefficients : ', regr.coef_)
print('intercept : ', regr.intercept_)

# print(regr.predict(x_data[:5]))
# print(regr.predict(x_scaled_data[:10]))
# 스케일드된 값을 학습시켰기 때문에 예측을 할때도 스케일드된 값으로 해야한다.
print(regr.predict(X_test))

y_true = y_test # 원래 y 값
y_pred = regr.predict(X_test) # 예측값
print(((y_true - y_pred) ** 2).sum()/len(y_true)) # RMSE의 지표

print(x_data[:5].dot(regr.coef_.T) + regr.intercept_)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error # RMSE의 지표
y_true =y_test
y_hat = regr.predict(X_test)

print(r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat))

y_true=y_train
y_hat=regr.predict(X_train)

print(r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat))
