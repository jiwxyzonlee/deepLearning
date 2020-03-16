# padas_line02.py
# pandas로 선그래프 그리기

import pandas as pd
import matplotlib.pyplot as plt

# x축에 index 값으로 날짜 출력, y축에 Series 데이터 출력
s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              # 번호 대신 index 값 설정 + 날짜를 index로
              , index = pd.date_range('2020-01-01'
                                      # 10개의 날짜 만들어짐
                                    , periods = 10))
print(s)

# 선 그래프
s.plot()
plt.show()

# 격자 모양 추가
s.plot(grid = True)
plt.show()