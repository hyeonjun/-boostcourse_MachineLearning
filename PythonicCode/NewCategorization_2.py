# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
'This is the first document.',
'This is the second second document',
'And the third one.',
'Is this the first document?',
]
x = vectorizer.fit_transform(corpus)
print(x.toarray())
print(vectorizer.get_feature_names())
