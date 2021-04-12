# -*- coding:utf-8 -*-
# Gradient descent based learning
# 실제 값과 학습된 모델 예측치의 오차를 최소화
# 모델의 최적 parameter 찾기가 목적
# gradient dascent 활용
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 1) # x 값
f_x = x**2 # f(x) = x^2
print(x)
print(f_x)
plt.plot(x,f_x) # 그래프

x_new = 10
derivative = []
y = []
learng_rate = 0.1 # 내려갈 정도(알파)
for i in range(100):
    old_value = x_new
    # x_new = x_old - a * (2 * x_old) # 2 * x_old는 f_x를 미분한 2*x 임
    derivative.append(old_value - learng_rate * 2 * old_value) # 미분값에 저장
    x_new = old_value - learng_rate * 2 * old_value # 새로운 x값
    y.append(x_new ** 2) # 새로운 x값으로 제곱하여 데이너를 넣음 그것으로 점 찍을거임
print(derivative)
print(y)

plt.scatter(derivative, y)
# -------------------------------------------------------------------

# def sin_funcfion(x):
#     return x * np.sin(x**2)+1
# def derivitive_f(x):
#     return np.sin(x**2) + 2*(x**2) * np.cos(x**2)
# x = np.arange(-3,3,0.001)
# f_x = sin_funcfion(x)
#
# print(derivitive_f(3))
#
# x_new = 1
# derivative = []
# y = []
# learng_rate = 0.01
# for i in range(10000):
#     old_value = x_new
#     x_new = old_value - learng_rate * derivitive_f(old_value)
#     derivative.append(x_new)
#     y.append(sin_funcfion(x_new))
#
# plt.plot(x, f_x)
# plt.scatter(derivative, y)

plt.show()
