# bar_graph.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 한글 글꼴 설정 -  '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 회원 ID
member_IDs = ['m_01', 'm_02', 'm_03', 'm_04']

# 운동 시작 전 윗몸 일으키기 횟수
before_ex = [27, 35, 40, 33]

# 운동 한 달 후 윗몸 일으키기 횟수
after_ex = [30, 38, 42, 37]


## 1. 기본 값으로 막대 그래프 출력

# 회원 네명, 전체 데이터 수 8개
n_data = len(member_IDs)

# NumPy를 이용해 배열 생성 (0, 1, 2, 3)
index = np.arange(n_data)

# bar(x, y)에서 x = index, height = before_ex 로 지정
plt.bar(index
        , before_ex)
plt.show()


## 2. tick_label : x축 tick 라벨 설정 (회원 ID로 설정)
plt.bar(index
        , before_ex
        , tick_label = member_IDs)
plt.show()


## 3. color : 막대 그래프의 색깔 설정
colors=['r', 'g', 'b', 'm']
# m : magenta
plt.bar(index
        , before_ex
        , color = colors
        , tick_label = member_IDs)
plt.show()


# 4. width : 막대 그래프의 폭 설정 ( 기본값 : 0.8 )
plt.bar(index
        , before_ex
        , tick_label = member_IDs
        , width = 0.6)
plt.show()


# 5. plt.barh() : 수평 막대 그래프
colors=['r', 'g', 'b', 'm']
plt.barh(index
         , before_ex
         , color = colors
         , tick_label = member_IDs)
plt.show()


# 6. 최종 그래프 출력
barWidth = 0.4
# 운동 전 그래프
plt.bar(index
        , before_ex
        # c : 청록색(cyan)
        , color='c'
        , align='edge'
        , width = barWidth
        , label='before')

# 운동 후 그래프
plt.bar(index + barWidth
        , after_ex
        , color='m'
        , align='edge'
        , width = barWidth
        , label='after')

plt.xticks(index + barWidth, member_IDs)

plt.legend()
plt.xlabel('회원 ID')
plt.ylabel('윗몸일으키기 횟수')
plt.title('운동 시작 전과 후의 근지구력(복근) 변화 비교')

plt.show()