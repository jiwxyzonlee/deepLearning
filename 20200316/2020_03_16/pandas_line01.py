# pandas_line01.py
# padas로 선그래프 그리기

import pandas as pd
import matplotlib.pyplot as plt

# 시리즈 생성
s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(s)

s.plot()
# kind = 'line' 생략 시 기본값은 선 그래프
plt.show()