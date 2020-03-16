# pandas_hist.py
# 25명 학생들의 수학점수를 히스토그램으로 시각화

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 한글 글꼴 설정 - '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 25명의 수학 점수
math = [76, 82, 84, 83, 90
    , 86, 85, 92, 72, 71
    , 100, 87, 81, 76, 94
    , 78, 81, 60, 79, 69
    , 74, 87, 82, 68, 79]

# 데이터프레임 생성
df = pd.DataFrame(math
                       , columns = ['Student'])
print(df)

# histogram 그래프
# 옵션 bins는 계급의 개수를 의미하며 기본값은 10
math_hist = df.plot.hist(bins = 8, grid=True)

math_hist.set_xlabel("시험 점수")
math_hist.set_ylabel("도수(frequency)")

math_hist.set_title("수학 시험의 히스토그램")
plt.show()