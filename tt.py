from torchtext import data
from torchtext import datasets

# Data Setting
TEXT = data.Field(lower=True, batch_first=True)     
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(TEXT, LABEL)

'''
torchtext에서 Dataset을 만들기 위해서는 각 데이터의 구성요소가 어떤 속성인지 알려주는 Field를 설정해야 한다. 
예를 들어 텍스트 데이터는 보통 문장을 토큰으로 나누거나, 소문자로 바꾸는 등의 전처리가 필요.
TEXT = Field(lower=True, tokenize=...) 같은 식으로 설정하면, 나중에 데이터를 불러올 때 이 설정에 따라 자동으로 처리된다. 
Field는 데이터를 불러올 때 어떻게 다룰지를 미리 약속하는 역할

splits() 함수는 torchtext가 제공하는 간편한 데이터 분할 도구 >> 데이터 준비 과정을 자동화해주는 함수
train, test = IMDB.splits(TEXT, LABEL)처럼 쓰면 IMDb 영화 리뷰 데이터를 자동으로 다운로드하고, 
학습용(train)과 테스트용(test) 데이터셋으로 나눠줌. 이때 각각의 데이터셋은 우리가 설정한 Field 규칙(TEXT, LABEL)을 따르도록 전처리.
'''