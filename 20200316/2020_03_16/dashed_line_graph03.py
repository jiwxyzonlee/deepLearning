# dashed_line_graph03.py

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 5, 8, 9, 5, 3]

plt.figure(figsize = (10, 6))

plt.plot(x, y
         # 선 색깔
         , color = 'green'
         # 선 종류
         , linestyle = 'dashed'
         # 마커 종류
         , marker = 'o'
         # 마커 색깔
         , markerfacecolor = 'red'
         # 마커 크기
         , markersize = 12)

plt.show()