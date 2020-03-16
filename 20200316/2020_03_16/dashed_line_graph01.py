# dashed_line_graph01.py

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 5, 8, 9, 5, 3]

# 그래프 크기 설정
plt.figure(figsize = (10, 6))

# 그래프 선 색과 종류 설정(기본값은 파란 선 그래프)
plt.plot(x, y
         , color = 'green'
         , linestyle = 'dashed')
plt.show()