import os
import cv2

#label_dict = {"zi":0,"chou":1,"yin":2,"mao":3,"chen":4,"si":5,"wu":6,"wei":7,"shen":8,"you":9,"xu":10,"hai":11}
label_dict = {0:"zi",1:"chou",2:"yin",3:"mao",4:"chen",5:"si",6:"wu",7:"wei",8:"shen",9:"you",10:"xu",11:"hai"}

label = label_dict[0]
output_path = "../DataSet/"

try: 
    path = os.path.join(output_path, label)
    os.makedirs(path, exist_ok = True) 
    print("Directory '%s' created successfully" %'directory') 
except:
    pass

size = 256 # image size
count = 0
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        
        h, w, _ = frame.shape
        x, y = (w - size) // 2, (h - size) // 2
        img = cv2.rectangle(frame, (x-1, y-1), (x + size + 1, y + size + 1), (0, 0, 255), 1)
        roi = frame[y:y+size, x:x+size]
        
        cv2.imshow('frame', img)
    else:
        roi = None

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif (key == ord('p') or key == ord('c')) and (roi is not None):
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        path = output_path + label + "/" + str(count) + ".jpg"

        ret = cv2.imread(path)
        if ret is None:
            cv2.imwrite(path, roi)
            print("cap", str(count), "!")
        else:
            print(str(count), "already exists!")
        count += 1

cap.release()
cv2.destroyAllWindows()
