# dashed_line_graph02.py

# 점선 그래프
# 마커 표시

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 5, 8, 9, 5, 3]

plt.figure(figsize = (10, 6))

# 초록색 점선 그래프 및 마커 표시
plt.plot(x, y
         , color = 'green'
         , linestyle = 'dashed'
         , marker = 'o')
plt.show()

plt.plot(x, y
         , color = 'green'
         , linestyle = 'dashed'
         , marker = '^')
plt.show()