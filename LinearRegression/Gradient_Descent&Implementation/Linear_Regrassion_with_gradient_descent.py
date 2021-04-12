import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def gen_data(numPoints, bias, variance): # (데이터갯수, w0값, 임의값)
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight Line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i+bias) + random.uniform(0,1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = gen_data(100, 25, 10)
plt.plot(x[:, 1]+1, y, "ro")
# ------------------------------------------------------------------------------



# df = pd.read_csv("slr06.csv")
# # print(df.head())
# raw_X = df["X"].values.reshape(-1,1)
# # print(raw_X)
# y = df["Y"].values
# # print(y)1
#
# plt.figure(figsize=(10,5))
#
# # plt.show()
#
# print(raw_X[:5], y[:5])
# print(np.ones((len(raw_X),1))[:3])
#
# X = np.concatenate((np.ones((len(raw_X),1)),raw_X), axis=1)
# print(X[:5])
#
# w = np.random.normal((2,1))
# print(w)
#
# plt.figure(figsize=(10,5))
# y_predict = np.dot(X,w)
# plt.plot(raw_X,y,"o",alpha=0.5)
# plt.plot(raw_X,y_predict)
#
# plt.show()

# ------------------------------------------------------------------------------
def hypothesis_function(X, theta):
    return X.dot(theta)  # x = ^y
#
# h = hypothesis_function(X,w)
# print(h)
#
def cost_function(h, y):
    return (1/(2*len(y))) * np.sum((h-y)**2)
# print(cost_function(h,y))
#
def gradient_descent(X, y, w, alpha, iterations):
    theta = w
    m = len(y)

    theta_list = [theta.tolist()]
    cost = cost_function(hypothesis_function(X, theta), y)
    cost_list = [cost]

    for i in range(iterations):
        t0 = theta[0] - (alpha / m) * np.sum(np.dot(X, theta) - y)
        t1 = theta[1] - (alpha / m) * np.sum((np.dot(X, theta) - y) * X[:,1])
        theta = np.array([t0, t1])

        if i % 10 == 0:
            theta_list.append(theta.tolist())
            cost = cost_function(hypothesis_function(X, theta), y)
            cost_list.append(cost)

    return theta, theta_list, cost_list

# --------------------------------------------------------------------------
# iterations = 10000
# alpha = 0.001
#
# theta, theta_list, cost_list = gradient_descent(X,y,w,alpha,iterations)
# cost = cost_function(hypothesis_function(X, theta), y)
#
# print("theta : ", theta)
# print("cost : ", cost_function(hypothesis_function(X, theta), y))
# print(theta_list[:10])
#
# theta_list = np.array(theta_list)
#
# print(cost_list[:5])
#
# plt.figure(figsize=(10,5))
# y_predict_step = np.dot(X, theta_list.transpose())
# print("y_predit_step : ", y_predict_step)
#
# plt.plot(raw_X,y,"o", alpha=0.5)
# for i in range(0,len(cost_list), 100):
#     plt.plot(raw_X, y_predict_step[:,i], label='Line %d' %i)
#
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# ------------------------------------------------------------------------

# plt.plot(range(len(cost_list)), cost_list)
# ------------------------------------------------------------------------

# th0 = theta_list[:, 0]
# th1 = theta_list[:, 1]
# TH0, TH1 = np.meshgrid(th0,th1)
#
# Js = np.array([cost_function(y, hypothesis_function(X, [th0, th1])) for th0, th1 in
#                zip(np.ravel(TH0), np.ravel(TH1))])
# Js = Js.reshape(TH0.shape)
#
# plt.figure(figsize=(8,4))
# CS = plt.contour(TH0, TH1, Js)
# plt.clabel(CS, inline=True, fontsize=10, inline_spacing=2)
# ------------------------------------------------------------------------

# from mpl_toolkits.mplot3d import Axes3D
#
# ms = np.linspace(theta[0] - 15, theta[0] + 15, 100)
# bs = np.linspace(theta[1] - 15, theta[1] + 15, 100)
# M,B = np.meshgrid(ms, bs)
#
# zs = np.array([ cost_function(y, hypothesis_function(X, theta))
#                 for theta in zip(np.ravel(M), np.ravel(B))])
# Z = zs.reshape(M.shape)
#
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.2)
# ax.contour(M,B,Z,10, color='b', alpha=0.5, offset=0, stride=30)
#
# ax.set_xlabel('Intercept')
# ax.set_ylabel('Slope')
# ax.set_zlabel('Cost')
# ax.view_init(elev=30., azim=30)
# ax.plot([theta[0]], [theta[1]], [cost_list[-1]], markerfacecolor='r', markeredgecolor='r',
#         marker='x', markersize=7)
# ax.plot(theta_list[:,0], theta_list[:,1], cost_list, markerfacecolor='g', markeredgecolor='g',
#         marker='o', markersize=1)
# ax.plot(theta_list[:,0], theta_list[:,1], 0, markerfacecolor='b', markeredgecolor='b',
#         marker='.', markersize=2)


plt.show()
