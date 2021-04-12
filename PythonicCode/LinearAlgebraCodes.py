# -*- coding:utf-8 -*-
##########################################################################################
# vector
##########################################################################################
u = [2,2]
v = [2,3]
z = [3,5]
result = []

# 아래와 같이 파이썬은 사용하면 안된다
for i in range(2):
    result.append(u[i]+v[i]+z[i])
print(result)

result = [sum(t) for t in zip(u,v,z)]
print(result)

u = [1,2,3]
v = [4,4,4]
alpha = 2
result = [alpha*sum(i) for i in zip(u,v)]
print(result)

##########################################################################################
# Matrix의 계산 : Matrix addition
# C = A + B = [ 3   6  ] + [ 5   8  ] = [ 8    14 ]
#               4   5        6   7        10   12
##########################################################################################
matrix_a = [[3,6], [4,5]]
matrix_b = [[5,8], [6,7]]
result = [[sum(row) for row in zip(*t)] for t in zip(matrix_a, matrix_b)]
# for t in zip(matrix_a, matrix_b)를 하면 tuple형으로
# ([3, 6], [5, 8]), ([4, 5], [6, 7]) 묶어진다
# *t를 사용하여 unpack이 되면서 각각의 변수로 풀려 [3,6] [5,8]을 가져온다
# 즉   [3,6]
#      [5,8] 이렇게 따로 되면서 zip을 사용하여 같은 인덱스 위치인 3과5,
# 6와 8이 sum에 의해 더해진다.
print(result)

##########################################################################################
# Matrix의 계산 : Scalar-Matrix Product
# a * A = 4 * [ 3   6  ]  = [ 12  24 ]
#               4   5         16  20
##########################################################################################
matrix_a = [[3,6],[4,5]]
alpha = 4
result = [[alpha*element for element in t] for t in matrix_a]
print(result)

##########################################################################################
# Matrix의 계산 : Matrix Transpose
#                        1  4
# A = [ 1 2 3 ], A^t = [ 2  5 ]
#       4 5 6            3  6
##########################################################################################
matrix_a = [[1,2,3], [4,5,6]]
result = [[element for element in t] for t in zip(*matrix_a)]
# *matrix_a를 사용하므로써 두개의 변수로 풀림
# [1, 2, 3]
# [4, 5, 6] 에서 같은 인덱스 위치인 [1,4] [2,5] [3,6]이 되어 t에 들어가게 된다.
# 그 후 element에 t의 값들이 각각 들어가 [[1,4],[2,5],[3,6]]
print(result)

##########################################################################################
# Matrix의 계산 : Matrix Product
#                        1  4                              1  1
# A = [ 1 2 3 ],   B = [ 2  5 ] => C = A * B = [ 1 1 2] * [2  1]=[ 5  8 ]
#       4 5 6            3  6                    2 1 1     1  3    5  6
#                                       => 1*1+1*2+2*1 = 5  | 1*1+1*1+2*3 = 8
#                                          2*1+1*2+1*1 = 5  | 2*1+1*1+1*3 = 6
##########################################################################################
matrix_a = [[1,1,2],[2,1,1]]
matrix_b = [[1,1],[2,1],[1,3]]
result=[[sum(a*b for a,b in zip(row_a, column_b)) \
 for column_b in zip(*matrix_b)] for row_a in matrix_a]
# *matrix_b => [1,4] [2,5] [3,6] 이여서 zip으로 묶어주어 1,2,3과 4,5,6 이 column_b에 들어간다.
print(result)
