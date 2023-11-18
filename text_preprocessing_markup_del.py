# 텍스트분석 전처리 과정
# 불필요한 특수 문자 및 마크업 문자 제거하는 함수

import re
def preprocessor(text):
    text=re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                          text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
           ' '.join(emoticons).replace('-', ''))
    return text
