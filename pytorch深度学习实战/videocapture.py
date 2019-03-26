#视频中抽取图片
import cv2
cap = cv2.VideoCapture("images/test.mp4")
success,frame = cap.read()
index = 1
while success :
    index = index+1
    cv2.imwrite(str(index)+".png",frame)
    if index >20:
        break;
    success,frame = cap.read()
cap.release()


















