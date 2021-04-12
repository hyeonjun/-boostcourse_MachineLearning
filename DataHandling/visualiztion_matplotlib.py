#--coding:utf-8--
import matplotlib.pyplot as plt
import numpy as np

# matplotlib
# pyplot 객체를 사용하여 데이터를 표시
# pyplot 객체에 그래프들을 쌓은 다음 show로 flush
# 최대 단점 argument를 kwarngs 받음, 고정된 argument가 없어서 alt+tab으로 확인어려움

# pandas matplotlib
# Pandas 0.7 버전 이상부터 matplotlib를 사용한 그래프 지원
# Dataframe, series 별로 그래프 작성 가능

# x = range(100)
# y = [value**2 for value in x]
#
# y = range(100)
# plt.plot(x,y)
#
# x_1 = range(100)
# y_1 = [np.cos(value) for value in x]
# x_2 = range(100)
# y_2 = [np.sin(value) for value in x]
# plt.plot(x_1, y_1)
# plt.plot(x_2, y_2)


# fig = plt.figure() # figure 반환
# fig.set_size_inches(10,10) # 크기 지정
# ax_1 = fig.add_subplot(1,2,1) # add_subplot : plot 생성
# ax_2 = fig.add_subplot(1,2,2) #
# ax_1.plot(x_1, y_1, c="b") # 첫번째 plot
# ax_2.plot(x_2, y_2, c="g") # 두번째 plot
# print("\n-----------------------------------------------")

# set color
# x_1 = range(100)
# y_1 = [value for value in range(100)]
# x_2 = range(100)
# y_2 = [value + 100 for value in range(100)]
# plt.plot(x_1, y_1, color="#000000")
# plt.plot(x_2, y_2, c="c")
# print("\n-----------------------------------------------")

# set linestyle
#         # c = color  /  linestyle = ls
# plt.plot(x_1, y_1, c="b", linestyle="dashed") # 긴 점선
# plt.plot(x_2, y_2, c="r", ls="dotted") # 짧은 점선

# set title
# plt.title("Two lines")
# print("\n")

# plt.title('$y = \\frac{ax + b}{test}$')
# print("\n")

# plt.title('$y = ax + b$')
# plt.xlabel('$x_line$')
# plt.ylabel('$y_line$')
# print("\n")

# plt.text(50,70, "Line_1")
# plt.annotate('line_2', xy=(50,150), xytext=(20, 175), arrowprops=dict(facecolor='black', shrink=0.05))
#           표시할 문자열  가리키는 위치   텍스트위치           화살표              색상은 블랙    굵기
# print("\n")


# print("\n-----------------------------------------------")
# plt.plot(x_1, y_1, c="b", linestyle="dashed", label='line_1') # 긴 점선
# plt.plot(x_2, y_2, c="r", ls="dotted", label="line_2") # 짧은 점선

# set legend
# legend 함수로 범례를 표시함, loc 위치 등 속성 지정
#             그림자      박스            위치
# plt.legend(shadow=True, fancybox=False, loc="upper right")

# plt.legend(shadow=True, fancybox=True, loc="lower right")
# plt.grid(True, lw=0.4, ls="--", c=".90")
# plt.xlim(-1000,2000)
# plt.ylim(-1000,2000)

# plt.grid(True, lw=0.4, ls="--", c=".90")
# plt.legend(shadow=True, fancybox=True, loc="lower right")
# plt.xlim(-100,200)
# plt.ylim(-200,200)
# plt.savefig("test.png", c="a") # 저장
# print("\n-----------------------------------------------")

# Scatter
# scatter 함수 사용, marker:scatter 모양 지정
# data_1 = np.random.rand(512, 2)
# data_2 = np.random.rand(512, 2)
#                                # c -> color / b -> blue
# plt.scatter(data_1[:,0], data_1[:,1], c="b",marker="x")
# plt.scatter(data_2[:,0], data_2[:,1], c="r",marker="o")
# print("\n")

# N=50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = np.pi * (15 * np.random.rand(N))**2
#                 s : 데이터의 크기를 지정, 데이터의 크기 비교가능
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# print("\n-----------------------------------------------")

#Bar chart
# data = [[5., 25., 50., 20.],
#         [4., 23., 51., 17],
#         [6., 22., 52., 19]]
# x = np.arange(0,8,2)
#           + 숫자는 위치               바의 넓이
# plt.bar(x + 0.00, data[0], color='b', width=0.50)
# plt.bar(x + 0.50, data[1], color='g', width=0.50)
# plt.bar(x + 1.00, data[2], color='r', width=0.50)
#
# plt.xticks(x+0.50, ("A","B","C","D"))
# print("\n")

# data = np.array([[5., 25., 50., 20.],
#                  [4., 23., 51., 17],
#                  [6., 22., 52., 19]])

# color_list = ['b', 'g', 'r']
# data_label = ["A","B","C"]
# x = np.arange(data.shape[1])
# # x = array([0,1,2,3])
#
# data = np.array([[5., 5., 5., 5.,],
#                  [4., 23., 51., 17],
#                  [6., 22., 52., 19]])
# for i in range(3):
#                          bottom : 위로 쌓기
#     plt.bar(x, data[i], bottom=np.sum(data[:i], axis=0),
#             color=color_list[i], label=data_label[i])
# plt.legend()
# print("\n")

# A = [5., 30., 45., 22.]
# B = [5, 25, 50, 20]
# x = range(4)
# plt.bar(x, A, color = 'b')
#                         #  아래 여백?
# plt.bar(x, B, color = 'r', bottom=60)
# print("\n")

# women_pop = np.array([5, 30, 45, 22])
# men_pop = np.array([5, 25, 50, 20])
# x = np.arange(4)
# # barh는 가로로 데이터 나옴
# plt.barh(x, women_pop, color='r')
# plt.barh(x, -men_pop, color='b')
# print("\n")

# x = np.arange(1000)
# plt.hist(x, bins=5)
x = np.random.randn(1000)
plt.hist(x, bins=100)
# print("\n")

# data = np.random.randn(100,5)
# plt.boxplot(data)

plt.show()
