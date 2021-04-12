# -*- coding:utf-8 -*-
import numpy as np

print("=======================================================")
print("numpy_ndarray")
print("=======================================================")
test_array = np.array([1,4,5,8], float)
print(test_array)
print(type(test_array[3]))
print("\n=======================================================\n")

test_array = np.array([1, 4, 5, "8"], np.float32) # String Type의 데이터를 입력해도
print(test_array)
print(type(test_array[3])) # Float Type으로 자동 형변환 실시
print(test_array.dtype) # Array(배열) 전체의 데이터 Type을 반환
print(test_array.shape) # Array의 object의 dimension 구성을 반환 => (4,) (row,)
print("\n=======================================================\n")

matrix = [[1,2,5,8],[1,2,5,8],[1,2,5,8]]
print(np.array(matrix, int).shape) # (3,4) (colum, row)
print("\n=======================================================\n")

tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
print(np.array(tensor, int).shape) # (4,3,4) (depth, row, colum)
# ndim - number of dimension
# size - count of data
print(np.array(tensor, int).ndim)
print(np.array(tensor, int).size)


print("\n=======================================================")
print("numpy_reshape")
print("=======================================================")
# Array의 shape의 크기를 변경 -> 2차원을 vector로 변경하는 등
test_matrix = [[1,2,3,4],[1,2,5,8]]
print(np.array(test_matrix).shape)
print(np.array(test_matrix).reshape(2,2,2)) # 깊이, 가로, 세로
print(np.array(test_matrix).reshape(8,))
print(np.array(test_matrix).reshape(-1,2)) # column을 2로 만들어줘야하는데 row 정확히 모를때 사용

print("\n=======================================================\n")
test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
print(np.array(test_matrix).flatten()) # 다차원의 Array를 1차원 array로 변환

print("\n=======================================================")
print("indexing_slicing")
print("=======================================================")

print("indexing")
test_example = np.array([[1,2,3], [4.5, 5,6]], int)
print(test_example)
print(test_example[0][0]) # 1
print(test_example[0,0]) # 1
test_example[0,0] = 12 # 0,0에 12 할당
print(test_example)
print("\n=======================================================\n")

print("slicing")
# List와 달리 행과 열 부분을 나눠서 slicing이 가능함
# Matrix의 부분 집합을 추출할 때 유용함
a = np.array([[1,2,3,4,5],[6,7,8,9,10]], int)
print(a[:2,:]) # Row - 0~1까지, Column - 전체 / [[1 2 3 4 5]
               #                                [6 7 8 9 10]]
print(a[1,1:3]) # Row - 1, Coulmn - 1~2 / [7 8]
print(a[1:3]) #  Row - 1~2의 전체 [[ 6 7 8 9 10]]
print("\n=======================================================\n")


print("\n=======================================================")
print("numpy_creation_functions")
print("=======================================================")
print(np.arange(30)) # range : List의 range와 같은 효과, intege로 0부터 29까지 배열추출
print(np.arange(0, 5, 0.5))
print(np.arange(0, 5, 0.5).tolist()) # floating point도 표시가능
            #시작, 끝, step
print(np.arange(30).reshape(5,6))
print(np.arange(30).reshape(-1,5))
a = np.arange(100).reshape(10,10)
print(a[:, -1].reshape(-1,1))
print("\n=======================================================\n")

print(np.zeros(shape=(10,), dtype=np.int8)) # 10 - zero vector 생성
print(np.zeros((2,5))) # 2 by 5 - zero matrix 생성
print(np.ones(shape=(10,), dtype=np.int8)) # 10 - one vector 생성
print(np.ones((2,5))) # 2 by 5 - zero matrix 생성
print(np.empty(shape=(10,), dtype=np.int8))
# empty - shape만 주어지고 비어있는 ndarray 생성(memory initialization이 되지 않음)
print(np.empty((3,5))) # 2 by 5 - zero matrix 생성
print("\n=======================================================\n")

test_matrix = np.arange(30).reshape(5,6)
print(np.ones_like(test_matrix)) # 기존 ndarray의 shape 크기만큼 1,0 또는 empty로 채움
print("\n=======================================================\n")

# identity - 단위 행렬(i 행렬)을 생성함(n -> number of rows)
print(np.identity(n=3, dtype=np.int8))
print(np.identity(5))
print("\n=======================================================\n")

# eye - 대각선인 1인 행렬, k값의 시작 index의 변경이 가능
print(np.eye(N=3, M=5, dtype=np.int8))
print(np.eye(3))
print(np.eye(3,5, k=2)) # k->start index
print("\n=======================================================\n")

# diag - 대각 행렬의 값을 추출
matrix = np.arange(9).reshape(3,3)
print(np.diag(matrix))
print(np.diag(matrix, k=1)) # k->start index

# random sampling - 데이터 분포에 따른 sampling으로 array 생성
print(np.random.uniform(0,1,10).reshape(2,5))
print(np.random.normal(0,1,10).reshape(2,5))

print("\n=======================================================")
print("numpy_operation_functions")
print("=======================================================")
# sum
test_array = np.arange(1,11)
print(test_array.sum(dtype=np.float))
print("\n=======================================================\n")

# axis - 모든 operation function을 실행할 때, 기준이 되는 dimension 축
test_array = np.arange(1,13).reshape(3,4)
#      axis =1(기준 : 가로)
# array([[1,2,3,4],      axis=0(먼저생긴것, 기준 : 세로)
#        [5,6,7,8]
#        [9,10,11,12]])
print(test_array.sum(axis=1), test_array.sum(axis=0))
# (array([10, 26, 42]), array([15, 18, 21, 24]))

third_order_tensor = np.array([test_array, test_array, test_array])
# array([[[ 1,  2,  3,  4],   깊이 -> 0, 가로 -> 1, 세로 -> 2
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]],
#        [[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]],
#        [[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]]])
print(third_order_tensor.sum(axis=2))
print(third_order_tensor.sum(axis=1))
print(third_order_tensor.sum(axis=0))
print("\n=======================================================\n")

test_array = np.arange(1,13).reshape(3,4)
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])
print(test_array.mean(), test_array.mean(axis=0))
print(test_array.std(), test_array.std(axis=0))
print(np.exp(test_array), np.sqrt(test_array))
print("\n=======================================================\n")

# concatenate - Numpy array를 합치는 함수
a = np.array([1,2,3])
b = np.array([2,3,4])
print(np.vstack((a,b))) # 수평 스택
a = np.array([ [1], [2], [3]])
b = np.array([ [2], [3], [4]])
print(np.hstack((a,b))) # 수직 스택
print("\n=======================================================\n")

a = np.array([[1,2,3]])  # [[1 2 3]
b = np.array([[2,3,4]])  #  [2 3 4]]
print(np.concatenate((a,b), axis=0))
a = np.array([[1,2],[3,4]])  # [[1 2 5]
b = np.array([[5,6]])        #  [3 4 6]]
print(np.concatenate((a,b.T), axis=1)) # T=> transpose
print(a.tolist())

print("\n=======================================================")
print("array_operations")
print("=======================================================")
# Numpy는 array간의 기본적인 사칙 연산을 지원함
test_a = np.array([[1,2,3],[4,5,6]], float)
print(test_a + test_a)
print(test_a*2 - test_a)
print("\n=======================================================\n")

# Element - wise array_operations
# Array간 shape이 같을 때 일어나는 연산
matrix_a = np.arange(1,13).reshape(3,4)
print(matrix_a * matrix_a)
print("\n=======================================================\n")

# Dot product
# Matrix의 기본 연산, dot 함수 사용
test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)
print(test_a.dot(test_b)) # 원래 행렬연산
print("\n=======================================================\n")

# transpose - transpose 또는 T attribute 사용
test_a = np.arange(1,7).reshape(2,3)
print(test_a.transpose())
print(test_a.T)
print(test_a.T.dot(test_a))
print("\n=======================================================\n")

# broadcasting - Shape이 다른 배열 간 연산을 지원하는 기능
test_matrix = np.array([[1,2,3],[4,5,6]], float)
scalar = 3
print(test_matrix + scalar) # 각 배열마다 더해줌
print("\n")
print(test_matrix - scalar)
print("\n")
print(test_matrix * scalar)
print("\n")
print(test_matrix / scalar)
print("\n")
print(test_matrix // 0.2)
print("\n")
print(test_matrix ** 2)
print("\n")
test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10)
print(test_matrix)
print("\n")
print(test_vector)
print("\n")
print(test_matrix + test_vector) # 각 자리마다 더하는거

print("\n=======================================================")
print("numpy_comparison")
print("=======================================================")
# All & Any
# Array의 데이터 전부(and) 또는 일부(or)가 조건에 만족 여부 반환
a= np.arange(10)
print(a>5)
print(np.any(a>5), np.any(a<0))
print(np.all(a>5), np.all(a<10))
print("\n=======================================================\n")

# logical_and, logical_not, logical_or
a = np.array([1,3,0], float)
print(np.logical_and(a>0, a<3)) # 두 가지 종류의 조건에 대해 모두 만족 여부 반환
b = np.array([True, False, True], bool)
print(np.logical_not(b)) # 반대(NOT 조건)
c = np.array([False, True, False], bool)
print(np.logical_or(b, c)) # OR조건
print("\n=======================================================\n")

# where
print(np.where(a> 0,3,2)) # where(condition, TRUE면, FALSE면)
a = np.arange(10)
print(np.where(a>0)) # index 값 반환
a = np.arange(5,15)
print(a)
print(np.where(a>10))
print("\n=======================================================\n")

a = np.array([1, np.NaN, np.Inf], float)
print(np.isnan(a)) # Not a Number
print(np.isfinite(a)) # is finite number
print("\n=======================================================\n")

a = np.array([1,2,4,5,8,78,23,3]) # 최대 최소값을 index 값으로 반환
print(np.argmax(a) , np.argmin(a))

a=np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
print(np.argmax(a, axis=1) , np.argmin(a, axis=0))

print("\n=======================================================")
print("boolean_fancy_index")
print("=======================================================")
# boolean index
# numpy는 배열은 특정 조건에 따른 값을 배열 형태로 추출할 수 있음
# Comparison operation 함수들도 모두 사용가능
test_array = np.array([1, 4, 0, 2, 3, 8, 9, 7], float)
print(test_array>3)
print(test_array[test_array > 3]) # 조건이 True인 index의 element만 추출
condition = test_array < 3
print(test_array[condition])
print("\n=======================================================\n")

A = np.array([
[12, 13, 14, 12, 16, 14, 11, 10,  9],
[11, 14, 12, 15, 15, 16, 10, 12, 11],
[10, 12, 12, 15, 14, 16, 10, 12, 12],
[ 9, 11, 16, 15, 14, 16, 15, 12, 10],
[12, 11, 16, 14, 10, 12, 16, 12, 13],
[10, 15, 16, 14, 14, 14, 16, 15, 12],
[13, 17, 14, 10, 14, 11, 14, 15, 10],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 17, 19, 16, 17, 18, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 12, 14, 11, 12, 14, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11]])
B = A < 15
print(B.astype(np.int)) # True -1, False -0
print("\n=======================================================\n")

# fancy index
# numpy는 array를 index value로 사용해서 값을 추출하는 방법
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer로 선언
print(a[b]) # bracket index, b 배열의 값을 index로 하여 a의 값들을 추출함
print(a.take(b)) # take 함수 : bracket index와 같은 효과
# index 0 1 2 3
#       2 4 6 8
print("\n")
a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
print(a[b,c]) # b를 row index, c를 column index로 변환하여 표시함
a = np.array([[1, 4], [9, 16]], float)
print(a[b])

print("\n=======================================================")
print("numpy_data_io")
print("=======================================================")
# loadtxt & savetxt
# Text type의 데이터를 읽고, 저장하는 기능
a = np.loadtxt("./populations.txt") # 파일 호출
print(a[:10])
print("\n")
a_int = a.astype(int)
print(a_int[:3])
np.savetxt('int_data.csv', a_int, delimiter=",")
print("\n=======================================================\n")

# Numpy object(pickle) 형태로 데이터를 저장하고 불러옴
# Binary 파일 형태로 저장함
np.save("npy_test", arr=a_int)
npy_array = np.load(file="npy_test.npy")
print(npy_array[:3])
