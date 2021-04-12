#--coding:utf-8--
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터 간의 상관관계를 볼 때 scatter graph 사용 가능
fig = plt.figure()
ax = []
for i in range(1,5):
    ax.append(fig.add_subplot(2,2,i))
# ax[0].scatter(df_data["CRIM"], df_data["MEDV"])
# ax[1].scatter(df_data["PTRATIO"], df_data["MEDV"])
# ax[2].scatter(df_data["AGE"], df_data["MEDV"])
# ax[3].scatter(df_data["NOX"], df_data["MEDV"])


ax[0].scatter(df_data["CRIM"], df_data["MEDV"], color="b", label="CRIM")
ax[1].scatter(df_data["PTRATIO"], df_data["MEDV"], color="g")
ax[2].scatter(df_data["AGE"], df_data["MEDV"])
ax[3].scatter(df_data["NOX"], df_data["MEDV"])
#    그래프 끼리의 가로 여백  세로 여백
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].legend()
ax[0].set_title("CRIM")
plt.show()
