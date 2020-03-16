# line_graph01.py

import matplotlib.pyplot as plt

# 주로 list 데이터 삽입
plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

# x축 라벨
plt.xlabel('x-axis')

# y축 라벨
plt.ylabel('y-axis')

# 라벨을 한글로 설정할 경우 깨짐, 소스코드 상에서도 오류
#plt.xlabel('x축')

# 그래프 출력
plt.show()