# -*- coding:utf-8 -*-
#############################################################################################
# Split 함수
#############################################################################################
items = 'zero one two three'.split() # 빈칸을 기준으로 문자열 나눠서 리스트로 만듬
print(items)

items = 'python,jquery,javascript'
result=items.split(",") # ,를 기준으로 나눠서 리스트만듬
print(result)
a,b,c = items.split(",") # ,를 기준으로 나눈 것들을 a,b,c 각 변수에 넣음
print(a+"//"+b+"//"+c)

#############################################################################################
# Join 함수
#############################################################################################
colors=["a","b","c","d","e"]
result = "".join(colors) # 빈칸없이 합치기
print(result)
result = " ".join(colors) # 빈칸 1칸으로 연결
print(result)

#############################################################################################
# List comprehensions
# - 기존 List 사용하여 간단히 다른 List를 만드는 기법
# - 포괄적인 List, 포함되는 리스트라는 의미로 사용됨
# - 파이썬에서 가장 많이 사용되는 기법 중 하나
# - 일반적으로 for + append 보다 속도가 빠름
#############################################################################################
result = [i for i in range(10)] # 0~9까지 첫번째 i에 i값이 순서대로 들어가면서
print(result)                   # result는 리스트 형식으로 i값을 받음

result = [i for i in range(10) if i % 2 == 0] # 0~9의 숫자 중 2의 배수만 i값으로 넣음
print(result)

word_1 = "Hello"
word_2 = "World"
result = [i+j for i in word_1 for j in word_2] # 여기에도 if문 추가 가능
print(result)
result = [[i+j for i in word_1] for j in word_2]
print(result)
result = [[i+j for j in word_2] for i in word_1]
print(result)

words = "the quick brown fox jumps over the lazy dog".split()
stuff = [[w.upper(), w.lower(), len(w)] for w in words]
for i in stuff:
    print(i)

#############################################################################################
# Enumerate : list의 element를 추출할 때 번호를 붙여서 추출 가능
#############################################################################################
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)

mylist = ["a", "b", "c", "d"]
print(list(enumerate(mylist))) # list에 있는 index와 값을 unpacking하여 list로 저장

# 문장을 빈큰으로 나누어 list로 만든 후 list의 각 index와 값을 unpacking하여 dict{}로 저장
print({i: j for i,j in enumerate('My name is Ju Hyeon Jun'.split())})

#############################################################################################
# Zip : 두개의 list의 값을 병렬적으로 추출
#############################################################################################
alist=['a1','a2','a3']
blist=['b1','b2','b3']
for a,b in zip(alist,blist): # 병렬적으로 값을 추출
    print(a,b)

# 각 tuple의 값은 index끼리 묶는다
a,b,c = zip((1,2,3), (10,20,30), (100,200,300))
print(a,b,c)

# 각 Tuple 값는 index를 묶어 합을 list로 변환시킴
print([sum(x) for x in zip((1,2,3), (10,20,30), (100,200,300))])

# Enumerate & zip
for i, (a,b) in enumerate(zip(alist, blist)):
    print (i,a,b)  # tuple로 출력

#############################################################################################
# Lambda
# - 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수
# - 수학의 람다 대수에서 유래함
#############################################################################################
f = lambda x,y : x+y
print(f(1,4)) # f가 함수처럼 쓰임
f = lambda x : x**2
print(f(3))
print((lambda x: x+1)(5)) # 5를 x에 대입

#############################################################################################
# Map Funtion : Sequence 자료형 각 element에 동일한 function을 적용함
#############################################################################################
ex =[1,2,3,4,5]
f = lambda x : x**2
print(list(map(f,ex))) # list를 꼭 붙여야하며, 안붙이려면 for문을 사용해야함

print([x ** 2 for x in ex]) # 위와 동일한 결과값이 나옴

f = lambda x,y: x+y
print(list(map(f,ex,ex))) # 결과값 : [2,4,6,8,10]

print(list(map(lambda x: x**2 if x%2 ==0 else x, ex))) # 아니라면 x에 원래 값 넣어라

#############################################################################################
# Reduce funtion : map function과 달리 list에 똑같은 함수를 적용해서 통합
#############################################################################################
from functools import reduce
print(reduce(lambda x,y: x+y, [1,2,3,4,5]))
    # 1+2=3 / 3+3=6 / 6+4=10 / 10+5=15 / 결과값 : 15

def factorial(n):
    return reduce(lambda x,y: x*y, range(1,n+1))
print(factorial(5))

#############################################################################################
# Collections
# List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조(모듈)
# 편의성, 실행 효율 등을 사용자에게 제공함
# from collections import deque
# from collections import Counter
# from collections import OrderedDict
# from collections import defaultdict
# from collections import namedtuple
#############################################################################################

#############################################################################################
# deque
# Stack과 Queue를 지원하는 모듈
# List에 비해 효율적인 자료 저장 방식을 지원함
#############################################################################################
from collections import deque
deque_list = deque()
for i in range(5):
    deque_list.append(i) # 뒤로 삽입
print(deque_list)
deque_list.appendleft(10) # 첫번째 위치에 삽입
print(deque_list)

# rotate, reverse 등 Linked List의 특정을 지원함
# 기존 list 형태의 함수를 모두 지원함
deque_list.rotate(2) # rotate : 회전, 두번 회전해서 10,0,1,2,3,4 -> 4,10,0,1,2,3 -> 3,4,10,0,1,2
print(deque_list)
deque_list.rotate(2)
print(deque_list)

print(deque(reversed(deque_list)))

deque_list.extend([5,6,7]) # 1,2,3 이였다면 1,2,3,5,6,7
print(deque_list)

deque_list.extendleft([5,6,7]) # 1,2,3 이였다면 7,6,5,1,2,3
print(deque_list)

# deque는 기존 list보다 효율적인 자료구조를 제공
# 효율적 메모리 구조로 처리 속도 향상
import time
start_time = time.clock()
deque_list = deque()
# Stack
# for i in range(1000):
#     for i in range(1000):
#         deque_list.append(i)
#         deque_list.pop()
# print(time.clock() - start_time, "seconds")
# 결과 : 0.2167521

# General list 방식
# start_time = time.clock()
# just_list=[]
# for i in range(1000):
#     for i in range(1000):
#         just_list.append(i)
#         just_list.pop()
# print(time.clock() - start_time, "seconds")
# 결과 : 0.503117499

#############################################################################################
# OrderedDict
# Dict와 달리, 데이터를 입력한 순서대로 dict를 반환함
#############################################################################################
from collections import OrderedDict
d={}
d['x']=100
d['y']=200
d['z']=300
d['l']=500
for k,v in d.items(): # 순서 뒤죽박
    print(k,v)

print "\n"

d = OrderedDict()
d['x']=100
d['y']=200
d['z']=300
d['l']=500
for k,v in d.items(): # 입력한 순서대로
    print(k, v)
print "\n"
# key -> ex) x, 100 이므로 x 자리가 index = 0임. 그래서 l, x, y, z 순으로 출력
for k, v in OrderedDict(sorted(d.items(), key=lambda t: t[0])).items():
    print(k, v)
print "\n"
# key -> 100, 200, 300, 500에서 reverse를 True했으므로 500 300 200 100
for k, v in OrderedDict(sorted(d.items(), reverse=True, key=lambda t: t[1])).items():
    print(k, v)
print "\n"

#############################################################################################
# defaultdick : Dick type의 값에 기본 값을 지정, 신규값 생성시 사용하는 방법
#############################################################################################
from collections import defaultdict
d = defaultdict(object) # Default dictionary를 생성
d = defaultdict(lambda : 0) # Defualt 값을 0으로 설정
print(d["first"]) # 값이 없지만, Default가 0이므로 0을 출력함

text = """A press release is the quickest and easiest way to get free publicity. If well written, a press release can result in multiple published articles about your firm and its products. And that can mean new prospects contacting you asking you to sell to them. Talk about low-hanging fruit!
What's more, press releases are cost effective. If the release results in an article that (for instance) appears to recommend your firm or your product, that article is more likely to drive prospects to contact you than a comparable paid advertisement.
However, most press releases never accomplish that. Most press releases are just spray and pray. Nobody reads them, least of all the reporters and editors for whom they're intended. Worst case, a badly-written press release simply makes your firm look clueless and stupid.
For example, a while back I received a press release containing the following sentence: "Release 6.0 doubles the level of functionality available, providing organizations of all sizes with a fast-to-deploy, highly robust, and easy-to-use solution to better acquire, retain, and serve customers."
Translation: "The new release does more stuff." Why the extra verbiage? As I explained in the post "Why Marketers Speak Biz Blab", the BS words are simply a way to try to make something unimportant seem important. And, let's face it, a 6.0 release of a product probably isn't all that important.
As a reporter, my immediate response to that press release was that it's not important because it expended an entire sentence saying absolutely nothing. And I assumed (probably rightly) that the company's marketing team was a bunch of idiots.""".lower().split()

print(text)
# dict 쓸 때
# word_count = {}
# for word in text:
#     if word in word_count.keys():
#         word_count[word] += 1
#     else:
#         word_count[word] = 0
# print(word_count)

# defaultdict 쓸 때
word_count=defaultdict(object)
word_count=defaultdict(lambda: 0)
for word in text:
    word_count[word] += 1
for i, v in OrderedDict(sorted(word_count.items(), key=lambda t: t[1], reverse=True)).items():
    print(i,v)


#############################################################################################
# Counter : Sequence type의 data element들의 갯수를 dict 형태로 반환
#############################################################################################
from collections import Counter
c = Counter()
c = Counter('gallagad')
print(c)
