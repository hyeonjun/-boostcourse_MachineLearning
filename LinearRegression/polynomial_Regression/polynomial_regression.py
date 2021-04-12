# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def f(size):
    x = np.linspace(0,5,size) # 0부터 5까지 size만큼(개수) 만들어라
    y = x * np.sin(x**2) + 1
    return (x,y)

def sample(size):
    x = np.linspace(0,5,size)
    rand = np.random.randn(x.size)*0.5
    # print(rand)
    y = x * np.sin(x**2) + 1 + rand
    return (x,y)

f_x, f_y = f(1000)
X, y = sample(1000)
# plt.plot(f_x, f_y)
# plt.scatter(X, y, s=3, c="black")
# ---------------------------------------------------------------------

# print(X.shape, y.shape) # ((1000,) , (1000,))

X = X.reshape(-1, 1) # 2 dimension(만지원하므로)
y = y.reshape(-1, 1) # 으로 들어가져야하기 때문에 (-1,1)로 reshape

# print(X.shape, y.shape) # ((1000, 1), (1000, 1))
#
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
# plt.plot(f_x, f_y)
# plt.scatter(X.flatten(), y.flatten(), s=3, c="black")
# plt.plot(X.flatten(), lr.predict(X).flatten())
# plot으로 들어갈때는 1 dimension으로 들어가야하기 때문에
# 2 dimension으로 되어있는 X의 값을 flatten시켜줌
# ---------------------------------------------------------------------


from sklearn.preprocessing import PolynomialFeatures

# poly_features = PolynomialFeatures(degree=2) # 2차 방정식으로 만들어줌
# X_poly = poly_features.fit_transform(X) # x1만 있었는데 x0, x1, x1^2값을 만들어줌
# print(X_poly[:10])
lr = LinearRegression(fit_intercept=False)
# PolynomialFeatures을 하게 되면 절편값이 만들어지는데 이것을 없애줘야함
# 그래서 fitting을 시킬때 없애기 위해 LinearRegression(fit_intercept=False)를 넣어줘야함
# lr.fit(X_poly, y)
#
# plt.plot(f_x, f_y)
# plt.scatter(X.flatten(), y.flatten(), s=3, c="black")
# plt.plot(X.flatten(), lr.predict(X_poly).flatten())
# ---------------------------------------------------------------------


# poly_features = PolynomialFeatures(degree=9)
# X_poly = poly_features.fit_transform(X)
# print(X_poly[:3])
# lr.fit(X_poly,y)
#
# plt.plot(f_x, f_y)
# plt.scatter(X.flatten(), y.flatten(), s=3, c="black")
# plt.plot(X.flatten(), lr.predict(X_poly).flatten())
# ---------------------------------------------------------------------


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
#
# poly_range : 얼마나 PolynomialFeatures을 만들것인가 결정
poly_range = list(range(10,50)) # 10~49까지 전부 넣는다 -> full search
rmse_lr_list = []
rmse_lasso_list = []
rmse_ridge_list = []
#
from sklearn.linear_model import Lasso # L1
from sklearn.linear_model import Ridge # L2
#
for poly_value in poly_range:
    poly_features = PolynomialFeatures(degree=poly_value) # feature 생성
    X_poly = poly_features.fit_transform(X)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_poly, y)

    # 값을 예측하는 모델을 만들고나서 모델에 대해 predict해줌
    # 그것을 rmse에 넣어주고 rmse_lr_list에 각각의 모델들마다 값을 저장시킴
    rmse_lr_list.append(rmse(lr.predict(X_poly), y))

    lasso = Lasso(fit_intercept=False)
    lasso.fit(X_poly, y)
    rmse_lasso_list.append(rmse(lasso.predict(X_poly), y))

    ridge = Ridge(fit_intercept=False)
    ridge.fit(X_poly, y)
    rmse_ridge_list.append(rmse(ridge.predict(X_poly), y))
#

# ---------------------------------------------------------------
# rmse_lr_list.append(rmse(lr.predict(X_poly), y))를
# 아래 부분에서 간단하게 볼 수 있다.
# import pandas as pd
# from pandas import DataFrame
# data = {"poly_range":poly_range, "lr_rmse":rmse_lr_list, # 전부 list 타입으로 되어있음
#         "lasso_rmse":rmse_lasso_list, "ridge_rmse":rmse_ridge_list}
# df = DataFrame(data).set_index("poly_range")
# print(df)
# ----------------------------------------------------------------
#
# plt.plot(poly_range, df["ridge_rmse"], label="ridge")
# plt.plot(poly_range, df["lr_rmse"], label="lr") # lr과 ridge는 좀 오르락내리락함
# plt.plot(poly_range, df["lasso_rmse"], label="lasso") # lasso는 좀 유지되고
# plt.legend()
# lasso(L1)가 유지되는 이유는 Sparse solution 특징으로, 모델에 많이 영향을 주지 않는 값들은
# 0으로 만듬. 즉, 의미없는 feature들을 0으로 만들어 안쓰이게되면서 유지되는 값이 나옴
#
# print(df.min()) # ridge_rmse가 제일 작은 값을 가짐
# print("\n")
# print(df["ridge_rmse"].sort_values().head()) # 작은 순서대로 sorting- 22일때 가장 작음
# print("\n")
#
poly_features  = PolynomialFeatures(degree=22) # 그래서 차수로 22넣어줌
X_poly = poly_features.fit_transform(X)
ridge = Ridge(fit_intercept=False)
ridge.fit(X_poly, y)
#
plt.plot(f_x, f_y)
plt.scatter(X.flatten(), y.flatten(), s=3, c="black")
plt.plot(X.flatten(), ridge.predict(X_poly).flatten())



# ---------------------------------------------------------------------
# df = pd.read_csv("yield.csv", sep="\t")
# print(df.head())
#
# plt.scatter(df["Temp"], df["Yield"])
plt.show()
