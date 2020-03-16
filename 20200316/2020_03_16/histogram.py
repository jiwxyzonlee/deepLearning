# histogram.py
# 25명 학생들의 수학점수를 히스토그램으로 시각화

import matplotlib.pyplot as plt
import matplotlib

# '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 25명 학생들의 수학성적
math = [76, 82, 84, 83, 90, 86, 85, 92, 72, 71, 100, 87, 81, 76, 94, 78, 81, 60, 79, 69, 74, 87, 82, 68, 79]

# 히스토그램 기본값(10개의 계급으로 나눠서 출력됨)
plt.hist(math)
plt.show()

# 60에서 100까지 계급의 크기를 5로 설정하여 8개의 계급으로 나눠서 출력
plt.hist(math, bins = 8)

plt.xlabel('시험 점수')
plt.ylabel('도수(frequency)')
plt.title('수학 시험의 히스토그램')
# 격자 모양 효과 주기
plt.grid()

plt.show()