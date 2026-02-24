from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
# BASE_PATH = "people-counter/assets/"
BASE_PATH = "assets/"
IMAGES_PATH = BASE_PATH+"images/"
VIDEOS_PATH = BASE_PATH+"videos/"
WEIGHTS_PATH = BASE_PATH+"Yolo-Weights/"
MASK_PATH = BASE_PATH+"mask/"


# Setting up video frame
cap = cv2.VideoCapture(VIDEOS_PATH+'15007393-hd_1920_1080_24fps.mp4')
cap.set(3,1280)
cap.set(4,720)

# inialising constants
total_count=[]
yolo_classes = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

# Loading model
model = YOLO(WEIGHTS_PATH+"yolov8l.pt")

mask = cv2.imread(MASK_PATH+'elevator-mask.png')

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# line for calculating
limits_up = [200,230,650,330]
limits_down = [650,400,1200,500]

while(True):
    sucess, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    
    imgGraphics = cv2.imread(IMAGES_PATH+"car-counter-graphic.png", cv2.IMREAD_UNCHANGED)

    print(imgGraphics.shape, img.shape)
    # img = cvzone.overlayPNG(img,imgGraphics,(0,0))
    h, w, _ = imgGraphics.shape
    img[0:h, 0:w] = imgGraphics
    
    results = model(imgRegion, stream=True)
    detections =np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # With opencv
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            # if want to trace the yolo bounding box
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255), thickness=1)

            # with cvzone

            # Bounding box
            bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)
            # cvzone.cornerRect(img, bbox, )
            
            # Confidence
            conf_score = math.ceil(box.conf[0]*100)/100

            cls = box.cls[0]
            curr_class =  yolo_classes[int(cls)]
            if curr_class in ["person"] and conf_score > 0.2: 
                # if want to show the class of object detected
                # cvzone.putTextRect(img,f'{conf_score, curr_class}',(max(0,x1),max(y1,35)), scale=0.6, thickness=1, offset=2)
                curr_arr = np.array([x1,y1,x2,y2,conf_score])
                detections = np.vstack((detections,curr_arr))


    line_up = cv2.line(img, (limits_up[0],limits_up[1]), (limits_up[2], limits_up[3]), (0,0,255), 5)
    line_down = cv2.line(img, (limits_down[0],limits_down[1]), (limits_down[2], limits_down[3]), (0,0,255), 5)

    results_tracker = tracker.update(detections)

    for result in results_tracker:
        # print(result)
        x1, y1, x2, y2, _id = result
        x1,y1,x2,y2,_id = int(x1),int(y1),int(x2),int(y2), int(_id)

        # Printing id over the bounding box
        bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)
        
        cv2.rectangle(img, (x1,y1), (x2,y2),(255,0,0), thickness=2)
        # cvzone.putTextRect(img, f'{_id}', (max(x1,0),max(y1,35)), scale=2, thickness=6, offset=10, colorR = (255,255,0))
        
        # Centre of object detected
        cx,cy = (x1+x2)//2,(y1+y2)//2
        # cv2.circle(img,(cx,cy),6,(144,144,255),cv2.FILLED) # if want to check if the circle deted is correct or not

        # Counting if it crosses the line
        if limits_up[0]<cx<limits_up[2] and limits_up[1]-30<cy<limits_up[1]+30 :
            cv2.line(img, (limits_up[0],limits_up[1]), (limits_up[2], limits_up[3]), (0,255,0), 5)
            if total_count.count(_id)==0:
                total_count.append(_id)
        
        if limits_down[0]<cx<limits_down[2] and limits_down[1]-30<cy<limits_down[1]+30 :
            cv2.line(img, (limits_down[0],limits_down[1]), (limits_down[2], limits_down[3]), (0,255,0), 5)
            if total_count.count(_id)==0:
                total_count.append(_id)


    cvzone.putTextRect(img, f"{len(total_count)}", (50,33), scale=2, thickness=4, offset=3, colorR = (205,199,149))



    cv2.imshow("Image", img)
    # cv2.imshow("MaskedImage", imgRegion)


    # if cv2.waitKey(0)==27:
    #     # For escaping and cleaning the memory
    #     break
    cv2.waitKey(1)

    

