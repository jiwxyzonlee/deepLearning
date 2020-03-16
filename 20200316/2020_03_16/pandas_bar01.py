# pandas_bar01.py
# 데이터프레임으로 그래프 그리기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 세트 만들기 (3개의 난수로 이루어진 10개의 배열)
# 난수 발생 : 10행 3열 구조
data_set = np.random.rand(10, 3)
print(data_set)

# pandas의 데이터프레임(DataFrame)생성
df = pd.DataFrame(data_set
                  , columns = ['A', 'B', 'C'])
print(df)

# 수직 막대 그래프
df.plot(kind = 'bar')
plt.show()

# 수평 막대 그래프 horizontal
df.plot(kind = 'barh')
plt.show()

# 영역 그래프 area
df.plot(kind = 'area')
plt.show()

# 영역 그래프 (겹치지 않게 나타내기)
df.plot(kind = 'area', stacked = False)
plt.show()