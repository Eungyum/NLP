# 사전 작업
# 로컬 컴퓨터에서 사용할 경우 JDK 설치 필요
# https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html
# 설치 경로를 환경 변수에 등록해야함

!pip install konlpy soynlp

import konlpy
import pandas as pd
import numpy as np

# 파일 로딩 및 분할
# 파일 출처 : https://github.com/e9t/nsmc
df_train = pd.read_csv('ratings_train.txt',
                       delimiter='\t', keep_default_na=False)

X_train = df_train['document'].values
y_train = df_train['label'].values

df_test = pd.read_csv('ratings_test.txt',
                     delimiter='\t', keep_default_na=False)

X_test = df_test['document'].valus
y_test = df_test['label'].values

print(len(X_train), np.bincount(y_train))
print(len(X_test), np.bincount(y_test))


# Okt 로딩 및 테스트
from konlpy.tag import Okt
okt = Okt()
print(X_train[4])
print(okt.morphs(X_train[4]))

from sklearn.fature_extraction.text import TfidfVectorizer

tfidf = TfidVectorizer(ngram_range=(1,2),
                       min_df=3,
                       max_df=0.9,
                       tokenizer=okt.morphs,
                       token_pattern=None)
tfidf.fit(X_train)
X_train_okt = tfidf.transform(X_train)
X_test_okt = tfidf.transform(X_test)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils.fixes import loguniform

sgd = SGDClassifier(loss='log', random_state=1)
param_dist = {'alpha': loguniform(0.0001, 100.0)}
rsv_okt = RandomizedSearchCV(estimator=sgd,
                            param_distributions=param_dist,
                            n_iter=50,
                            random_state=1,
                            verbose=1)
rsv_okt.fit(X_train_okt, y_train)

print(rsv_okt.best_score_)
print(rsv_okt.best_params_)

rsv_okt.score(X_test_okt, y_test)
