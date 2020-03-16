# line_graph03.py

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

# plt.plot(x, y)

# 빨간색 점
plt.plot(x, y, 'ro')
plt.show()

# 빨간색 선 그래프
plt.plot(x, y, 'r-')
plt.show()

plt.plot(x, y, 'rv')
plt.show()

plt.plot(x, y, 'r>')
plt.show()

plt.plot(x, y, 'r<')
plt.show()

plt.plot(x, y, 'r^')
plt.show()

# 'r-', 'g-', 'b-' : 선 그래프
# 'ro', 'go', 'bo' : (x, y)점에 'o' 마크
 # 'rv', 'gv', 'bv' : (x, y)점에 'v' 마크(역삼각형)
 # 'r>', 'g>', 'b>' 도 가능 (오른쪽 표시 삼각형)
# plot()은 b- 옵션이 기본값 : 파란색(b) 라인(-)이라는 뜻
# ro 옵션은 빨간색(r)으로 o표시를 의미함
# bv 옵션은 파란색(b)으로 v표시를 의미함