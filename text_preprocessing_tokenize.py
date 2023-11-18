# 문장을 토큰 단위로 나누는 함수
# tokenizer() : 공백을 기준으로 나눔
# tokenizer_porter() : 공백을 기준으로 나눈 후, 어간추출 및 적용

from nltk.stem.porter import PorterStemmer

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
