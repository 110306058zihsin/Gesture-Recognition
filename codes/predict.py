import cv2
import copy
import json
import random
import sys, os
import operator
from keras.models import load_model
from keras.models import model_from_json

# File Path
gesture_imgs_path = r"frame_effect/gestures/"


# Dictionary
gesture_labels = {"zi":0,"chou":1,"yin":2,"mao":3,"chen":4,"si":5,"wu":6,"wei":7,"shen":8,"you":9,"xu":10,"hai":11, "nosign": 12}
pattern_dict = {"CHIDORI":["chou", "mao", "shen"], "GOGAKYUNOJYUTSU":["si", "wei", "shen", "hai", "wu", "yin"]}

# Load Gesture Images
gesture_imgs = {}
for key in gesture_labels.keys():
    img = cv2.imread(gesture_imgs_path + key + ".png") 
    h, w, _ = img.shape
    for i in range(6):
        intensity = 51 * i
        cv2.rectangle(img, (i, i), (w - 1 - i, h - 1 - i), (intensity, intensity, intensity) , 1)
    gesture_imgs[key] = cv2.resize(img, (0, 0), fx=0.4, fy=0.4) 


def getROI(roi, wzs=120):
    ROI = cv2.resize(roi, (128, 128))
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY) # ROI 轉灰階
    _, thresh = cv2.threshold(ROI, wzs, 255, cv2.THRESH_TOZERO)
    return thresh

def stichPatternImgs(pattern):
    if pattern == []: return None
    pattern_img = gesture_imgs[pattern[0]]
    for key in pattern[1:]:
        pattern_img = cv2.hconcat([pattern_img, gesture_imgs[key]])
    gray_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return pattern_img, gray_img

def addEffect(frame, effect, skill_name):
    frame_height, frame_width, _ = frame.shape
    effect = cv2.resize(effect, (frame_width, frame_height))
    gray_effect = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    
    _, mask1 = cv2.threshold(gray_effect, 1, 255, cv2.THRESH_BINARY)
    mask2 = cv2.bitwise_not(mask1)
    masked_effect = cv2.bitwise_and(effect, effect, mask=mask1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask2)

    effected_frame = cv2.add(masked_effect, masked_frame)

    font_size = 2
    font_style = cv2.FONT_HERSHEY_DUPLEX
    text_size, _ = cv2.getTextSize(skill_name, font_style, font_size, 12)
    p = ((frame_width - text_size[0]) // 2, frame_height - text_size[1] - 2)

    for i in range(5):
        intensity = 51 * i
        cv2.putText(effected_frame, skill_name, p, font_style, font_size, (intensity , intensity, intensity), 11+i)
    cv2.putText(effected_frame, skill_name, p, font_style, font_size, (255, 102, 102), 10)
    cv2.putText(effected_frame, skill_name, p, font_style, font_size, (255, 0, 0), 8)
    return effected_frame

def addGestureHint(frame, pattern_img, pattern_gray_img):
    frame_height, frame_width, _ = frame.shape
    pattern_height, pattern_width, _ = pattern_img.shape
    x = (frame_width - pattern_width) // 2
    frame[:pattern_height, x:x + pattern_width] = pattern_img 
    frame[:pattern_height, x + (pattern_width - pattern_gray_img.shape[1]):x + pattern_width] = pattern_gray_img 

def addPredictSign(frame, predict):
    frame_height, frame_width, _ = frame.shape
    predict_height, predict_width, _ = predict.shape
    x = (frame_width - predict_width) // 2
    y = frame_height - predict_height - 50
    frame[y:y + predict_height, x:x + predict_width] = predict


# 讀訓練好的model
with open('model_trained2.json','r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_trained2.h5')


wzs = 120 # 二值化閥值
size = 256
cap = cv2.VideoCapture(0) # 擷取鏡頭畫面

success_count = 0

rand = 0 #random.randrange(len(pattern_dict.keys()))
skill_name = list(pattern_dict.keys())[rand]
target_pattern = pattern_dict[skill_name].copy()
hint_img, hint_gray_img = stichPatternImgs(target_pattern)

effect_file = "../" + skill_name + "/"
effect_filenames = []

start = False
clock = 0
while True:
    ret, FrameImage = cap.read() # 讀取鏡頭畫面
    
    if ret:
        FrameImage = cv2.flip(FrameImage, 1) # 圖像水平翻轉
        # 框出ROI位置
        h, w, _ = FrameImage.shape
        x, y = (w - size) // 2, (h - size) // 2
        cv2.rectangle(FrameImage, (x - 1, y - 1), (x + size + 1, y + size + 1), (0,255,0) ,1)

        if len(effect_filenames) > 0:
            effect_img = cv2.imread(effect_file + effect_filenames.pop(0))
            FrameImage = addEffect(FrameImage, effect_img, skill_name)
            addGestureHint(FrameImage, hint_img, hint_gray_img)
        else:

            if success_count == len(target_pattern):
                rand = 1 #random.randrange(len(pattern_dict.keys()))
                skill_name = list(pattern_dict.keys())[rand]
                target_pattern = pattern_dict[skill_name].copy()
                print(target_pattern)
                hint_img, hint_gray_img = stichPatternImgs(target_pattern)
                effect_file = "../" + skill_name + "/"
                success_count = 0
            
            addGestureHint(FrameImage, hint_img, hint_gray_img)

            if clock % 10 == 0:
                ROI = getROI(FrameImage[y:y+size, x:x+size])
                result = loaded_model.predict(ROI.reshape(1,128, 128, 1))

                predict = {}
                for key in gesture_labels.keys():
                    predict[key] = result[0][gesture_labels[key]]
                # 分數較高者會sort至第一位
                predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True) 
                predict_img = gesture_imgs[predict[0][0]]
                
                if predict[0][0] == target_pattern[success_count] and start:
                    hint_gray_img = hint_gray_img[::, hint_img.shape[1] // len(target_pattern):]
                    if len(hint_gray_img[0]) == 0:
                        effect_filenames = os.listdir(effect_file)
                    success_count += 1
            
            addPredictSign(FrameImage, predict_img)
   

        cv2.imshow("Frame", FrameImage) # 顯示鏡頭畫面

    clock += 1
    interrupt = cv2.waitKey(10)
    if interrupt == ord('l'): # lower wzs quality
        wzs = wzs - 5
    elif interrupt == ord('u'): # upper wzs quality
        wzs = wzs + 5
    elif interrupt == ord('a'): # esc key
        start = True
    elif interrupt == ord('q'): # esc key
        break
            
cap.release()
cv2.destroyAllWindows()
