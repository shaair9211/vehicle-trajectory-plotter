# Initialize an empty dictionary
new_dict = {}

# List of new state names
new_states = ['Michigan', 'Ohio', 'Georgia']

# Assign a common value (e.g., 6) to each new state
for state_name in new_states:
    new_dict[state_name] = [6, 4, 4, 6, 4]


length = len(new_dict[state_name])
print(length)
print("Dictionary with multiple keys and common value:")
print(new_dict)











from typing import Union, Any
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
import openpyxl

cap = cv2.VideoCapture("../Videos/3 way traffic.mp4")  # For Video
model = YOLO("../Yolo-Weights/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
mask = cv2.imread("Object Tracking/mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # creating instance
trajectories = {}
p = 0
q = 1
while True:
    success, img = cap.read()
    if not success:
        break
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf: float = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass: Union[str, Any] = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                cvzone.putTextRect(img, f' {currentClass}', (x1 + 10, y1 + 20), colorT=(255, 255, 255),
                                   colorR=(60, 20, 220), font=cv2.FONT_HERSHEY_TRIPLEX, scale=0.75, thickness=1,
                                   offset=2)
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=17, t=2, rt=1, colorR=(255, 228, 181), colorC=(60, 20, 220))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        print("cx,cy :", cx, cy)
        print("id:", id)

        if id in trajectories:
            trajectories[id].extend([cx, cy])
        else:
            trajectories.update({id: [cx, cy]})
        print("Trajectories:", trajectories)

        key_number = list(trajectories.keys())
        length_dictionary = len(trajectories[id])

        if length_dictionary >= 4:
            first_coordinate = trajectories[id][p]
            second_coordinate = trajectories[id][q]
            third_coordinate = trajectories[id][p + 2]
            fourth_coordinate = trajectories[id][q + 2]
            print("First coordinate:", first_coordinate)
            print("Second coordinate:", second_coordinate)
            print("Third coordinate:", third_coordinate)
            print("Fourth coordinate:", fourth_coordinate)
        else:
            pass
    cv2.imshow("Image", img)
    cv2.waitKey(0)
