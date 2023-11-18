# 불용어
# 분석에 무의미한 단어들

# 불용어 사전 다운로드
from nltk
nltk.download('stopwords')


# 불용어제거 사용 방법
from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

# 결과 : ['runner', 'like', 'run', 'run', 'lot']
