# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    # csv 타입 데이터 로드, separate는 빈공간으로 지정, column은 없음
    df = pd.read_csv(filename)
    # unstack : group으로 묶여진 데이터를 matrix 형태로 전환
    return df.groupby(["source", "target"])["rating"].sum().unstack().fillna(0)




def get_frequent_matrix(filename, dtype=np.float32):
    df = pd.read_csv(filename)
    df["rating"] = 1
    Z = df.groupby(["source", "target"])["rating"].sum().unstack().fillna(0)
    return Z.values


print(get_rating_matrix("movie_rating.csv"))
print(get_frequent_matrix("1000i.csv"))
