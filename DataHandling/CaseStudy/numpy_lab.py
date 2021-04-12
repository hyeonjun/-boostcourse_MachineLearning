# -*- coding:utf-8 -*-
import numpy as np

# 함수목적
#  - n의 제곱수로 2 dimentional array를 생성하는 ndarray.
# Args
#  - n: 생성하고자 하는 ndarray의 row와 column의 개수
#  - dtype: 생성하려는 ndarray의 data type (np.int)
# Returns
#  - row와 column의 길이가 n인 two dimentional ndarray로
#       X[0,0]은 0으로 순차적으로 X[n-1,n-1]은 n^2-1이 할당됨
# def n_size_ndarray_creation(n, dtype=np.int):
#     return np.array(range(n**2), dtype=dtype).reshape(n,n)
# print(n_size_ndarray_creation(3))


# 함수목적
#  - shape이 지정된 크기의 ndarray를 생성,
#     이때 행렬의 element는 type에 따라 0, 1 또는 empty로 생성됨.
# Args
#  - shape: 생성할려는 ndarray의 shape
#  - type: 생성되는 element들의 값을 지정함0은 0, 1은 1, 99는 empty 타입으로 생성됨
#  - dtype: 생성하려는 ndarray의 data type (np.int)
# Returns
#  - shape의 크기로 생성된 ndarray로 type에 따라 element의 내용이 변경됨
# def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
#     if type==0:
#         return np.zeros(shape=shape, dtype=dtype)
#     if type==1:
#         return np.ones(shape=shape, dtype=dtype)
#     if type==99:
#         return np.empty(shape=shape, dtype=dtype)
# print(zero_or_one_or_empty_ndarray((5,5)))



# 함수목적
#  - 입력된 ndarray X를 n_row의 값을 row의 개수로 지정한 matrix를 반환함.
#  - 이때 입력하는 X의 size는 2의 거듭제곱수로 전제함.
#  - 만약 n_row과 1일 때는 matrix가 아닌 vector로 반환함.
# Args
#  - X: 입력하는 ndarray
#  - n_row: 생성할려는 matrix의 row의 개수
# Returns
#  - row의 개수가 n_row인 Matrix 또는 Vector
#  - n_row가 1이면 Vector 값으로 반환함
# def change_shape_of_ndarray(X, n_row):
#     if n_row == 1:
#         return X.flatten()
#     else:
#         return X.reshape(n_row, -1)
# x_array = np.ones((2,2), dtype=np.int)
# print(change_shape_of_ndarray(x_array, 4))
# [[1]
#  [1]
#  [1]
#  [1]]


# 함수목적
# 입력된 ndarray X_1과 X_2를 axis로 입력된 축을 기준으로 통합하여 반환하는 함수
# X_1과 X_2는 matrix 또는 vector 임, 그러므로 vector 일 경우도 처리할 수 가 있어야 함
# axis를 기준으로 통합할 때, 통합이 불가능하면 False가 반환됨.
# 단 X_1과 X_2 Matrix, Vector 형태로 들어왔다면, Vector를 row가 1개인 Matrix로
# 변환하여 통합이 가능한지 확인할 것
# Args
# X_1: 입력하는 ndarray
# X_2: 입력하는 ndarray
# axis: 통합의 기준이 되는 축 0 또는 1임
# Returns
# X_1과 X_2과 통합된 matrix 타입의 ndarray
# def concat_ndarray(X_1, X_2, axis):
#     try:
#         if X_1.ndim == 1:
#             X_1 = X_1.reshape(1,-1)
#         if X_2.ndim == 1:
#             X_2 = X_2.reshape(1,-1)
#             print("trans :  {0}" .format(X_2))
#         return np.concatenate((X_1,X_2,),axis=axis)
#     except ValueError as e:
#         return False
# x_1 = np.array([[1,2,3],[7,8,9]]) # metrix인 경우
# # x_2 = np.array([[4,5,6]])
# x_2 = np.array([4,5,6]) # vertor인 경우 reshape하여 [[4,5,6]]으로 변환시켜 처리
# print(x_1.ndim)
# print(x_2.ndim)
# a = np.array([[1,2]])
# b = np.array([[5,6]])
# print(concat_ndarray(x_1, x_2, 0))
# print(concat_ndarray(a,b,1))

# 함수목적
# 입력된 Matrix 또는 Vector를 ndarray X의 정규화된 값으로 변환하여 반환함
# 이때 정규화 변환 공식 Z = (X - X의평균) / X의 표준편차로 구성됨.
# X의 평균과 표준편차는 axis를 기준으로 axis 별로 산출됨.
# Matrix의 경우 axis가 0 또는 1일 경우, row 또는 column별로 Z value를 산출함.
# axis가 99일 경우 전체 값에 대한 normalize 값을 구함.
# Args
# X: 입력하는 ndarray,
# axis: normalize를 구하는 기준이 되는 축으로 0, 1 또는 99임,
# 단 99는 axis 구분 없이 전체값으로 평균과 표준편차를 구함
# dtype: data type으로 np.float32로 구정
# Returns
# 정규화된 ndarray
# def normalize_ndarray(X, axis=0, dtype=np.float32):
#     X = X.astype(np.float32) # data type을 np.float32으로
#     n_row, n_column = X.shape # shape만 하면 해당 ndarray의 (row, column) 값이 나옴
#     print("("+str(n_row)+","+str(n_column)+")")
#     if axis == 99:
#         x_mean = np.mean(X) # 평균
#         print("mean : \n{0}" .format(x_mean))
#         x_std = np.std(X) # 표준편차
#         print("std : \n{0}" .format(x_std))
#     if axis == 1: # 가로별
#         x_mean = np.mean(X, 1)
#         print("mean : \n{0}" .format(x_mean))
#         x_mean = np.mean(X, 1).reshape(n_row, -1) # reshape하는 이유는 세로로 한줄 만들기 위해서
#         x_std = np.std(X,1).reshape(n_row, -1)
#         print("std : \n{0}" .format(x_std))
#     if axis == 0: # 세로별
#         x_mean = np.mean(X,0).reshape(1,-1)
#         print("mean : \n{0}" .format(x_mean))
#         x_std = np.std(X,0).reshape(1,-1)
#         print("std : \n{0}" .format(x_std))
#     Z = (X - x_mean) / x_std
#     return Z
# x = np.arange(12, dtype=np.float32).reshape(6,2)
# print(x)
# print(normalize_ndarray(x))


# 함수목적
# 입력된 ndarray X를 argument filename으로 저장함
# Args
# X: 입력하는 ndarray
# filename: 저장할려는 파일이름
# def save_ndarray(X, filename="test"):
#     return np.save(filename, arr=X)
# X = np.arange(32, dtype=np.float32).reshape(4,-1)
# save_ndarray(X)
# npy_array = np.load(file="test.npy")
# print(npy_array)


# 함수목적
# 입력된 ndarray X를 String type의 condition 정보를
# 바탕으로 해당 컨디션에 해당하는 ndarray X의 index 번호를 반환함
# 단 이때, str type의 조건인 condition을 코드로
# 변환하기 위해서는 eval(str("X") + condition)를 사용할 수 있음
# Args
# X: 입력하는 ndarray
# condition: string type의 조건 (">3", "== 5", "< 15")
# Returns
# 조건에 만족하는 ndarray X의 index
# def boolean_index(X, condition): # 내 답
#     return(X[eval(str("X")+condition)])
# def boolean_index(X, condition):
#     condition = eval(str("X") + condition)
#     return np.where(condition) # index 값 반환
# X = np.arange(32, dtype=np.float32)
# print(boolean_index(X,"%2==0"))

# 함수목적
# 입력된 vector type의 ndarray X에서 target_value와
# 가장 차이가 작게나는 element를 찾아 리턴함
# 이때 X를 list로 변경하여 처리하는 것은 실패로 간주함.
# Args
# X: 입력하는 vector type의 ndarray
# target_value : 가장 유사한 값의 기준값이 되는 값
# Returns
# target_value와 가장 유사한 값
# def find_nearest_value(X, target_value):
#     return X[np.argmin(np.abs(X-target_value))]
#     # argmin = 가장 작은 값의 index
#     # abs = 절대값으로 바꿈
# target_value = 0.3
# X = np.random.uniform(0,1,100)
# print(find_nearest_value(X, target_value))


# 함수목적
# 입력된 vector type의 ndarray X에서 큰 숫자 순서대로 n개의 값을 반환함.
# Args
# X: vector type의 ndarray
# n: 반환할려는 element의 개수
# Returns
# ndarray X의 element중 큰 숫자 순서대로 n개 값이 반환됨 ndarray
def get_n_largest_values(X, n):
    return X[np.argsort(X[::-1])[:n]]
    # sort해서 index를 가져옴
    # X[::]하면 전체를 다 보는데, -1을 붙이면 역순이기때문에
    # 원래 sort는 작은거부터 큰순, 역순으로하면 큰것부터 작은순으로 sort된다
    # index로 뽑았으니 X[index]로 가져오되 [:n]으로 갯수를 정한다.

X = np.random.uniform(0, 1, 5)
print(X)
# print(np.argsort(X[::-1])[:5])
print(get_n_largest_values(X, 3))
