# line_graph_label.py

import matplotlib.pyplot as plt
import matplotlib
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

# 한글 라벨 설정 - '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# x축 라벨(한글 라벨)
plt.xlabel('x축')

# y축 라벨 (한글 라벨 )
plt.ylabel('y축')

# x, y 데이터 값 두 개 모두 출력하므로
plt.plot(x, y)

plt.show()