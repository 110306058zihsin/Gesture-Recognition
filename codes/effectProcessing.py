import cv2
import numpy as np

save_path = "../GOGAKYUNOJYUTSU/"
cap = cv2.VideoCapture('../effect_videos/flame.mp4')

def adjustment1(frame, threshold):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b, g, r = frame[::, ::, 0], frame[::, ::, 1], frame[::, ::, 2]
    _, mask1 = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(r, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask1, mask2)
    img = cv2.bitwise_and(frame, frame, mask=mask)
    return img

def adjustment2(frame, threshold):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 0, 100]) 
    upper = np.array([100, 255, 255]) 
    mask = cv2.inRange(hsv, lower, upper) 
    mask = cv2.bitwise_not(mask)
    img = cv2.bitwise_and(frame, frame, mask=mask)
    return img

def adjustment(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(frame, frame, mask=mask)
    return img


threshold = 25
count = 0
while cap.isOpened() and count < 200:
    ret, frame = cap.read()
    if not ret: break

    img = cv2.flip(frame, 1) # 圖像水平翻轉
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = adjustment(img, threshold)
    cv2.imwrite(save_path + str(count) + ".png", img)
    cv2.imshow('frame', img)
    count += 1
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        threshold += 5
        print(threshold)
    elif key == ord('s'):
        threshold -= 5
        print(threshold)


cap.release()
cv2.destroyAllWindows()
