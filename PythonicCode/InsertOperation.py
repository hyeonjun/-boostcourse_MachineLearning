# -*- coding:utf-8 -*-
import itertools
from functools import reduce
import sys
import time

def insert_operation(length, input_num, input_oper):
    ops={"0":(lambda x,y: x+y), "1":(lambda x,y: x-y), "2":(lambda x,y: x*y), "3":(lambda x,y: x//y)}
    oper_permutation=[]
    result=[]
    list(oper_permutation.extend(
    [str(index)]*value) for index, value in enumerate(input_oper) if value>0)
    permutation = [list(x) for x in set(itertools.permutations(oper_permutation))]
    print(permutation)
    for i in permutation:
        result.append(reduce(lambda x,y : ops[i.pop()](x,y), input_num))
        # reduce : 여러 개의 데이터를 대상으로 주로 누적 집계를 내기 위해 사용
        # reduce(집계합수, 순회가능한데이터[, 초기값])
        print(lambda x,y:ops[i.pop()](x,y), input_num)
    print(str(max(result))+"\n"+str(min(result)))

n = 3
number = [1,2,3,4,5,6]
arithmetics=[2,1,1,1]
insert_operation(n, number, arithmetics)
