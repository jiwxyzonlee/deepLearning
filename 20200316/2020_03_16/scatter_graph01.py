# scatter_graph01.py

# 산점도 함수 scatter()

# 배열 생성 패키지
#import numpy as np

import matplotlib.pyplot as plt

#x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#y = np.array([9, 8, 7, 9, 8, 3, 2, 4, 3, 4])

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [9, 8, 7, 9, 8, 3, 2, 4, 3, 4]

plt.figure(figsize=(10, 6))

plt.scatter(x, y
            , alpha = 0.5
            , s = 50)

plt.show()