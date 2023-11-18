# Bow
## 언어를 알고리즘에 적용시키기 위해서는 수치 형태로 변환시켜야함
## Bow(Bag of Word) : 텍스트를 수치 특성 형태로 표현하는 모델
## Bow 아이디어
#   1. 전체 문서에 대해 고유한 토큰을 생성
#   2. 특정 문서에 각 단어가 얼마나 자주 등장하는지 헤아려 문서의 특성 벡터를 생성

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

# docs의 문장들에서 사용된 단어가 딕셔너리로 저장됨
print(count.vocabulary_)

# 특성 벡터 출력
print(bag.toarray())


# tf-idf를 사용하여 단어 적합성 평가
## 실제로 클래스 레이블이 다른 문서에 같은 단어들이 나타나는 경우가 많음
## 이런 단어들은 유용하거나 판별에 필요한 정보가 아님
## 그래서 이런 단어들의 가중치를 낮출 필요가 있음
## tf-idf(term frequency-inverse document frequency) : 단어 빈도와 역분서 빈도의 곱
## 사이킷런의 TfidfTransformer

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                       norm='l2',
                       smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
