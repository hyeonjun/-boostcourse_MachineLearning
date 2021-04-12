# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()
# print(boston.keys())
# print(boston.feature_names)

df = pd.DataFrame(boston.data, columns=boston.feature_names)
# print(df.head())

X = df.values
y = boston.target



# from sklearn.linear_model import  LinearRegression
# # LinearRegression - Normal equation
# lr_ne = LinearRegression(fit_intercept=True)
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# lr_ne.fit(X_train, y_train)
#
# y_hat = lr_ne.predict(X_test)
# y_true = y_test
#
# rmse = np.sqrt((((y_hat - y_true)**2).sum() / len(y_true)))
# print(rmse)
#
import sklearn
# mse = sklearn.metrics.mean_squared_error(y_hat, y_true)
# print(mse)
#
# plt.scatter(y_true, y_hat, s=10)
# plt.xlabel("Prices: $Y_i$")
# plt.ylabel("Predicted prices: $\hat{Y}_i$")
# plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
#
# print(lr_ne.coef_)
# print(boston.feature_names)

# ---------------------------------------------------------
# Linear Regressoin with SGD
# ---------------------------------------------------------
# from sklearn.linear_model import SGDRegressor
# # SGD의 단점 : 여러가지 hyperparameter를 사람이 직접 설정해줘야함
# # SGDRegressor(loss='squared_loss',penalty='l1 or l2', alpha=람다값,
# #           tol=멈추는기준, shuffle(섞기)=Ture,
# #           learning_rate=줄어드는전략(constant(결과를중간중간체크) or optimal or invscaling),
# #           eta0=learning_rate값(학습량빠르게->값올려))
#
# lr_SGD = SGDRegressor()
# # lr_SGD = SGDRegressor(n_iter=100000000, eta0=0.00001,learning_rate="constant")
#
# from sklearn.preprocessing import StandardScaler
# std_scaler = StandardScaler()
# std_scaler.fit(X)
# X_scaled = std_scaler.transform(X) # 스케일링
#
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#     test_size=0.33, random_state=42)
# # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
# #     test_size=0.33, random_state=42)
#
# lr_SGD.fit(X_train, y_train) # fitting 시킴
#
# y_hat = lr_SGD.predict(X_test)
# y_true = y_test
#
# mse = sklearn.metrics.mean_squared_error(y_hat, y_true)
# rmse = np.sqrt((((y_hat - y_true)**2).sum() / len(y_true)))
# print(rmse, mse)
# ### (101011995028310.45, 1.0203423139599416e+28)
# # rmse의 값이 엄청나게 큼 -> 학습이 제대로 안됌
# # train_test_split할때 X말고 X_scaled를 넣어주면 잘된다
# # 스캐일링으로 하기 싫다면 eta를 굉장히 작은값으로 하고 iteraion을 많이 돌린다.
#
#
# plt.scatter(y_true, y_hat, s=10)
# plt.xlabel("Prices: $Y_i$")
# plt.ylabel("Predicted prices: $\hat{Y}_i$")
# plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")


# ---------------------------------------------------------
# Linear Regressoin with Ridge & Lasso regression
# ---------------------------------------------------------
from sklearn.linear_model import Lasso, Ridge
# Lasso - L1 regression
# Ridge - L2 regression
# Ridge(solver는 어떤 알고리즘으로 학습을 시킨건지 정함, )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(X_train, y_train)
# lasso = Lasso(fit_intercept=True, alpha=0.5)

y_hat = ridge.predict(X_test)
y_true = y_test
mse = sklearn.metrics.mean_squared_error(y_hat, y_true)
rmse = np.sqrt((((y_hat - y_true)**2).sum() / len(y_true)))
print(rmse, mse)
### (4.579058484791169, 20.967776607137992)

from sklearn.model_selection import KFold
print('Ridge Regression')
print('alpha\t RMSE_train\t RMSE_10cv\n')
alpha = np.linspace(.01,20,50)
t_rmse = np.array([])
cv_rmse = np.array([])

for a in alpha:
    ridge = Ridge(fit_intercept=True, alpha=a)

    # computing the RMSE on training data
    ridge.fit(X_train,y_train)
    p = ridge.predict(X_test)
    err = p-y_test
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))

    # computin RMSE using 10-fold cross validation
    kf = KFold(10)
    xval_err = 0
    for train,test in kf.split(X):
        ridge.fit(X[train], y[train])
        p = ridge.predict(X[test])
        err = p - y[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(X))

    t_rmse = np.append(t_rmse, [rmse_train])
    cv_rmse = np.append(cv_rmse, [rmse_10cv])
    print('{:.3f}\t {:.4f}\t\t {:.4f}' .format(a,rmse_train,rmse_10cv))

plt.plot(alpha,t_rmse, label='RMSE-Train')
plt.plot(alpha,cv_rmse, label='RMSE_XVal')
plt.legend( ('RMSE-Train', 'RMSE_XVal'))
plt.ylabel('RMSE')
plt.xlabel('Alpha')
plt.show()

a = 0.3
for name, met in [
    ('linear regression', LinearRegression()),
    ('lasso', Lasso(fit_intercept=True, alpha=a)),
    ('ridge', Ridge(fit_intercept=True, alpha=a))
]:
    met.fit(X_train, y_train)
    # p = np.array([met.predict(xi) for xi in x])
    p = met.predict(X_test)
    e = p-y_test
    total_error = np.dot(e,e)
    rmse_train = np.sqrt(total_error/len(p))

    kf = KFold(10)
    err = 0
    for train,test in kf.split(X):
        met.fit(X[train], y[train])
        p = met.predict(X[test])
        e = p-y[test]
        err+=np.dot(e,e)

    rmse_10cv = np.sqrt(err/len(X))
    print('Method: %s' %name)
    print('RMSE on training: %.4f' %rmse_train)
    print('RMSE in 10-fold CV: %.4f' %rmse_10cv)
