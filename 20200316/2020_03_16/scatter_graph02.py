# scatter_graph02.py

# 산점도 colomap 기능


import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([9, 8 ,7, 9, 8, 3, 2, 4, 3, 4])

colormap = x

plt.figure(figsize=(10,6))

plt.scatter(x, y
            , s = 50
            , c = colormap
            , marker = '>')

plt.colorbar()

plt.show()