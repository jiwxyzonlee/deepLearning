import numpy as np
import cv2
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model('mnist_mlp_model.h5')        # 학습된 모델 불러오기
print("Loading complete!")

onDown = False
xprev, yprev = None, None

def onmouse(event, x, y, flags, params):            # 마우스 이벤트 처리 함수
    global onDown, img, xprev, yprev
    if event == cv2.EVENT_LBUTTONDOWN:              # 왼쪽 마우스 눌렀을 경우
        # print("DOWN : {0}, {1}".format(x,y))
        onDown = True
    elif event == cv2.EVENT_MOUSEMOVE:              # 마우스 움직일 경우
        if onDown == True:
            # print("MOVE : {0}, {1}".format(x,y))
            cv2.line(img, (xprev,yprev), (x,y), (255,255,255), 20)
    elif event == cv2.EVENT_LBUTTONUP:              # 왼쪽 마우스 눌렀다가 놓았을 경우
        # print("UP : {0}, {1}".format(x,y))
        onDown = False
    xprev, yprev = x,y

cv2.namedWindow("image")                        # 윈도우 창의 title
cv2.setMouseCallback("image", onmouse)          # onmouse() 함수 호출
width, height = 280, 280
img = np.zeros((280,280,3), np.uint8)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1)

    if key == ord('r'):                 # r 버튼 클릭 : clear
        img = np.zeros((280,280,3), np.uint8)
        print("Clear.")

    if key == ord('s'):                 # s 버튼 클릭 : 예측값 출력
        x_resize = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
        x_gray = cv2.cvtColor(x_resize, cv2.COLOR_BGR2GRAY)
        x = x_gray.reshape(1, 28*28)
        y = model.predict_classes(x)    # 모델에서 예측값 구해오기
        print(y)                        # 예측값 출력

    if key == ord('q'):                 # q 버튼 클릭 : 종료
        print("Good bye")
        break

cv2.destroyAllWindows()                 # 윈도우 종료