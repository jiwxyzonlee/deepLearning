# pandas_bar02.py

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 한글 글꼴 설정 - '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

grade_num = [5, 14, 12, 3]
students = ['A', 'B', 'C', 'D']

# DataFrame 생성
df = pd.DataFrame(grade_num)
print(df)

#  students로 인덱스 설정, columns 값 Student가 범례로 출력
df_grade = pd.DataFrame(grade_num
                        , index=students
                        # column 번호 0번 대신 Student(컬럼명)로
                        , columns=['Student'])
print(df_grade)

# 막대그래프
df_grade.plot.bar()
plt.show()

grade_bar = df_grade.plot.bar(grid = True)
# 격자무늬 배경 추가

# x축 라벨 설정
grade_bar.set_xlabel("학점")
# y축 라벨 설정
grade_bar.set_ylabel("학생수")
# 그래프 main title 설정
grade_bar.set_title("학점별 학생수 막대 그래프")

plt.show()