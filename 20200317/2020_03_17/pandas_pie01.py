# pandas_pie.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 5개의 난수로 이루어진 데이터 생성
data = np.random.rand(5)
print(data)

# pandas 시리즈형 데이터로 변환
s = pd.Series(data
              , index=['a', 'b', 'c', 'd', 'e']
              , name='series')
print(s)

# pie 그래프 출력
s.plot(
    # 그래프 종류 설정
    kind='pie'
    # 비율 표시, 소수점 제어
    , autopct='%.2f'
    # 크기 설정
    , figsize=(7, 7)
)
plt.show()