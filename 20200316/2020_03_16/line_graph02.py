# line_graph02.py

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 5, 8, 9, 5, 3]

# 그래프 크기 설정 (가로 10, 세로 6)
plt.figure(figsize = (10, 6))

# 그래프 선 색 변경 green
# 기본값 blue
plt.plot(x, y, color = 'green')

plt.show()