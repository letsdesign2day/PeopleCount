
import numpy as np
import cv2

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
cap = cv2.VideoCapture('Data/3.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) #width 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #height 
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness 

classNames = []
with open('Modal/coco.text','r') as f:
    classNames = f.read().splitlines()
#print(classNames)


Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "Modal/frozen_inference_graph.pb"
configPath = "Modal/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    ret,frame = cap.read()
    ClassIndex,confidence,bbox = net.detect(frame,confThreshold =0.60)
    count=0
    #print(ClassIndex)
    for x in ClassIndex:
        if ClassIndex[x]==1:
            count+=1
    #print(count)    
            
    if(len(ClassIndex)!=0):
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classNames[ClassInd-1],(boxes[0]+10,boxes[1]+40,),cv2.FONT_HERSHEY_COMPLEX, 1,color=(0,255,0),thickness=1)
                cv2.putText(frame,"Person in the frame :"+str(count),(50,30),cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0),2)
    cv2.imshow('object detection',frame)

    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()