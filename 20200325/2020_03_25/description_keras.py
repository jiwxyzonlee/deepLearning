 # keras
 # 케라스(keras)는 파이썬으로 구현된 간결한 딥러닝 라이브러리
 # 딥러닝 비전문가라도 각자가 필요한 분야에서 손쉽게 딥러닝 모델을 개발하고 활용핛 수 있도록
 # 케라스는 직관적인 API를 제공
 # 내부적으로는 텐서플로우(Tensorflow), 티아노(Theano), CNTK 등의 딥러닝 전용 엔진이 구동
 # 그러나 케라스 사용자는 복잡한 내부 엔진까지는 알 필요는 없음
 # 직관적인 API로 쉽게 다층퍼셉트론 신경망 모델, 컨볼루션 신경망 모델 또는 이를 조합한 모델은
 # 물론 다중 입력 또는 다중 출력 등 다양한 기능을 구현할 수 있음

#  keras의 주요 특징
#  파이썬 기반
#  - Caffe 처럼 별도의 모델 설정 파일이 필요 없으며 파이썬 코드로 모델들이 정의된다.
#  최소주의 (Minimalism)
#  - 각 모듈은 짧고 간결하다.
#  - 모든 코드는 한 번 훑어보는 것으로도 이해가 가능하다.
#
#  모듈화(Modularity)
#  - 케라스에서 제공되는 모듈은 독립적으로 설정 가능하며, 가능한 한 최소한의 제약사항으로 서로 연결 가능
#  모델은 시퀀스 또는 그래프로 이러한 모듈들을 구성한 것
#  - 특히 신경망 층, 비용함수, 최적화 기법, 활성화 함수, 정규화 기법은 모두 독립적인 모듈이며,
#  새로운 모델을 만들기 위해서는 이러한 모듈 조합가능
#  쉬운 확장성
#  - 새로운 클래스나 함수로 모듈을 아주 쉽게 추가 가능
#  - 고급 연구에 필요한 다양한 표현 가능

# tensorflow2.0 이 설치되면 keras도 같이 설치
# 그러나 keras에서 제공되는 모듈을 import 해서 사용하는 경로가 변경됨

# 기존 keras 에서 import 하는 방법
#  from keras.models import Sequential
#  from keras.layers import Dense
# Tensorflow2.0 에서 import 하는 방법
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense