# -*- coding:utf-8 -*-
# 한 뉴스에 대해 비슷한 뉴스를 찾아내도록 하기
# Process
# 파일을 불러오기
# 파일을 읽어서 단어사전(corpus) 만들기 - set을 사용하여 여러번 나와도 한번만 쓰이도록
# 단어별로 Index 만들기
# 만들어진 인덱스로 문서별로 Bag of words vector 생성
# 비교하고자 하는 문서 비교하기
# 얼마나 맞는지 측정하기
import os
def get_file_list(dir_name): # 파일 불러오기
    return os.listdir(dir_name)

def get_contents(file_list): # 파일별로 내용 읽기
    y_class=[] # 80개의 텍스트 파일 중 야구에 관한건지 축구에 관한건지 0과 1로 분류
    x_text=[]
    class_dick={1:"0", 2:"0", 3:"0", 4:"0", 5:"1", 6:"1", 7:"1", 8:"1"}
              # 0은 야구,                   1은 축구
    for file_name in file_list:
        try:
            f = open(file_name, "r")
            category = int(file_name.split(os.sep)[1].split("_")[0])
            y_class.append(class_dick[category])
            x_text.append(f.read())
            f.close()
        except UnicodeDecodeError as e:
            print(e)
            print(file_name)
    return x_text, y_class

def get_cleaned_text(text): # 의미없는 문장보호 등은 제거하기
    import re   # I'm yours -> imyours
    text = re.sub('\W+', '', text.lower() )
    return text

def get_corpus_dict(text):
    text = [sentence.split() for sentence in text] # 80개의 텍스트을 하나씩받아서 스플릿
    # text는 two dimension 리스트이다
    cleaned_words = [get_cleaned_text(word) for words in text for word in words]
    # 텍스트에서 단어들만 뽑아서, 단어들에서 단어를 가져온다.-> one dimension array

    from collections import OrderedDict
    corpus_dict = OrderedDict()
    for i,v in enumerate(set(cleaned_words)): # set : 같은 단어는 제거
        corpus_dict[v]=i    # enumerate : 인덱스값 가져옴
    return corpus_dict

def get_count_vector(text, corpus):
    text = [sentence.split() for sentence in text]
    word_number_list = [[corpus[get_cleaned_text(word)] for word in words] for words in text]
    # two dimension 형태로 만듬

    # 0으로 채워진 매트릭스 만듬
    x_vector = [[0 for _ in range(len(corpus))] for x in range(len(text))]
    # _ : 변수 사용하지 않는다는 말이며, 0으로 채운다는 의미

    for i, text in enumerate(word_number_list):
        for word_number in text:
            x_vector[i][word_number] += 1
    return x_vector

import math
def get_cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2 : (v1 dot v2)/{||v1||*||v2||}"
    sumxx, sumxy, sumyy = 0,0,0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def get_similarity_score(x_vector, source):
    source_vector = x_vector[source]
    similarity_list = []
    for target_vector in x_vector:
        similarity_list.append(get_cosine_similarity(source_vector, target_vector))
    return similarity_list

def get_top_n_similarity_news(similarity_score, n):
    import operator
    x = {i:v for i, v in enumerate(similarity_score)}
    # sorted : 키값으로 정렬
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))

    return list(reversed(sorted_x))[1:n+1]

def get_accuracy(similarity_list, y_class, source_news):
    source_class =y_class[source_news]

    return sum([source_class == y_class[i[0]] for i in similarity_list]) / len(similarity_list)

if __name__ == "__main__":
    dir_name = "news_data"
    file_list = get_file_list(dir_name)
    file_list = [os.path.join(dir_name, file_name) for file_name in file_list]

    x_text, y_class = get_contents(file_list)
    corpus = get_corpus_dict(x_text)
    x_vector = get_count_vector(x_text, corpus)

    source_number = 10
    result=[]

    for i in range(80):
        source_number = i

        similarity_score = get_similarity_score(x_vector, source_number)
        similarity_news = get_top_n_similarity_news(similarity_score, 10)
        accuracy_score = get_accuracy(similarity_news, y_class, source_number)
        result.append(accuracy_score)
    print(sum(result) / 80)
