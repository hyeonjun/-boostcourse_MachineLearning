# -*- coding:utf-8 -*-
from sklearn import datasets
from sklearn.linear_model import Lasso, Ridge
boston = datasets.load_boston()

X = boston.data
y = boston.target
# -------------------------------------------------------------------


from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    print("TRAIN - ", len(train_index))
    print("TEST - ", len(test_index))
# -------------------------------------------------------------------


from sklearn.metrics import mean_squared_error
kf = KFold(n_splits=10)
# lasso_regressor = Lasso()
# ridge_regressor = Ridge()
#
lasso_mse = []
ridge_mse = []
#
# for train_index, test_index in kf.split(X):
#     lasso_regressor.fit(X[train_index], y[train_index])
#     ridge_regressor.fir(X[train_index], y[train_index])
#     lasso_mse.append(mean_squared_error(y[test_index], lasso_regressor.predict(X[test_index])))
#     ridge_mse.append(mean_squared_error(y[test_index], lasso_regressor.predict(X[test_index])))
#
# print(sum(lasso_mse)/10, sum(ridge_mse)/10)
# -------------------------------------------------------------------


from sklearn.model_selection import cross_val_score
import numpy as np
# lasso_regressor = Lasso(warm_start=False)
# ridge_regressor = Ridge()
#
# lasso_scores = cross_val_score(lasso_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
# ridge_scores = cross_val_score(ridge_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
# print(np.mean(lasso_scores), np.mean(ridge_scores))
# -------------------------------------------------------------------


from sklearn.model_selection import cross_validate
# import numpy as np
# lasso_regressor = Lasso(warm_start=False)
# ridge_regressor = Ridge()

# scoring = ['neg_mean_squared_error', 'r2']
# lasso_scores = cross_validate(lasso_regressor, X, y, cv=10, scoring=scoring)
# ridge_scores = cross_validate(ridge_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
# print(lasso_scores)
# -------------------------------------------------------------------


from sklearn.model_selection import cross_val_score
import numpy as np
# lasso_regressor = Lasso(warm_start=False)
# ridge_regressor = Ridge()
#
# kf = KFold(n_splits=10, shuffle=True)
# lasso_scores = cross_val_score(lasso_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')
# ridge_scores = cross_val_score(ridge_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')
# print(np.mean(lasso_scores), np.mean(ridge_scores))
# -------------------------------------------------------------------


from sklearn.model_selection import LeaveOneOut
# test = [1,2,3,4]
# loo = LeaveOneOut()
# for train, test in loo.split(test):
#     print("%s %s" % (train,test))
#
# lasso_scores = cross_val_score(lasso_regressor, X, y, cv=loo, scoring='neg_mean_squared_error')
# ridge_scores = cross_val_score(ridge_regressor, X, y, cv=loo, scoring='neg_mean_squared_error')
# print(np.mean(lasso_scores), np.mean(ridge_scores))
#
# lasso_scores = cross_val_score(lasso_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')
# ridge_scores = cross_val_score(ridge_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')

import matplotlib.pyplot as plt
# labels=["LASSO", "RIDGE"]
# plt.boxplot((lasso_scores, ridge_scores), labels=labels)
# plt.grid(linestyle="--")
# -------------------------------------------------------------------


def rmse(predictions, targets):
    return np.sqrt(((predictions-targets) ** 2).mean())
#
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X)
X_scaled = std.transform(X)
# #
eta0 = 0.0001
max_iter = 100
#
from sklearn.model_selection import train_test_split
X_train_dataset, X_test, y_train_dataset, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train_dataset, X_test, y_train_dataset, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# sgd_regressor = SGDRegressor(
#     eta0=eta0, max_iter=max_iter, warm_start=True, learning_rate="constant"
# )
sgd_regressor = SGDRegressor(
    eta0=eta0, n_iter=max_iter, warm_start=True, learning_rate="constant"
)
rmse_val_score = []
rmse_train_score = []
model_list = []

X_train, X_val, y_train, y_val = train_test_split(
    X_train_dataset, y_train_dataset, test_size=0.2, random_state=42)
sgd_regressor.fit(X_train, y_train)
#
# # kf = KFold(n_split=100, shuffle=True
# # for train_index, test_index in kf.split(X_train_dataset):
for i in range(300):
    y_pred = sgd_regressor.predict(X_train)
    y_true = y_train
    rmse_train_score.append(rmse(y_pred, y_true))

    y_pred = sgd_regressor.predict(X_val)
    y_true = y_val
    rmse_val_score.append(rmse(y_pred, y_true)) # train과 validation 차이 보기
    model_list.append(sgd_regressor) # weight값들이 저장되어있음

    coef = sgd_regressor.coef_.copy() # 만든 모델의 coef를 카피해서 넣음
    intercept = sgd_regressor.intercept_.copy() # 절편값 카피해서 넣음

    sgd_regressor = SGDRegressor( # warm_start : 기존에 있는 값을 사용할것임
        eta0=eta0, max_iter=max_iter, warm_start=True, learning_rate="constant"
    )
    #                       100돌려서 만든 coef와 intercept를 넣어서 쓰자
    sgd_regressor.fit(X_train,y_train,coef_init=coef, intercept_init=intercept)
#
plt.plot(range(len(rmse_val_score)), rmse_val_score, c="G", label="VAL")
plt.plot(range(len(rmse_train_score)), rmse_train_score, c="r", label="TRAINING")
plt.scatter(99, rmse(y_test,sgd_regressor.predict(X_test)), s=1, label="TEST")
plt.legend()
#
plt.show()
# 엄청나게 튀는 경향이 있는데 왜 일까?
# 처음에 normalization을 해줘야 안정적인데 안해서 그런듯하다
# 무조건 해야하는 것도 아니고, 꼭 해야하는 것도 아님
#
print(np.argsort(rmse_val_score))
print(rmse(y_test,sgd_regressor.predict(X_test)))
print(rmse(y_test,model_list[217].predict(X_test)))
print(model_list[0].coef_)
