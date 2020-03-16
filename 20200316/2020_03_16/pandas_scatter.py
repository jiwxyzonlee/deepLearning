# pandas_scatter.py
# pandas로 산점도 그리기

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 한글 글꼴 설정 - '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

temperature = [25.2, 27.4, 22.9, 26.2, 29.5
    , 33.1, 30.4, 36.1, 34.4, 29.1]
Ice_cream_sales = [236500, 357500, 203500, 365200, 446600
    , 574200, 453200, 675400, 598400, 463100]

dict_data = {'기온' : temperature
    , '아이스크림 판매량' : Ice_cream_sales}

# 데이터 프레임 생성
df = pd.DataFrame(dict_data)
print(df)

df_ice_cream = pd.DataFrame(dict_data
                            , columns=['기온', '아이스크림 판매량'])
print(df_ice_cream)

# 산점도 그래프 그리기
df_ice_cream.plot.scatter(x='기온', y='아이스크림 판매량'
                          , grid = True
                          , title='최고 기온과 아이스크림 판매량')
plt.show()