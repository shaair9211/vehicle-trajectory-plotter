from typing import Union, Any
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
import openpyxl

cap = cv2.VideoCapture("Videos/demo_1.mp4")  # For Video
model = YOLO("YOLO-Weights/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
mask = cv2.imread("Object Tracking/mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # creating instance

def draw_trajectory(img, trajectories):
    for id, positions in trajectories.items():
        # Draw lines connecting consecutive positions
        for i in range(1, len(positions)):
                cv2.line(img, positions[i-1], positions[i], (0,206,209), 1)  #  color, line thickness 1

# Initialize dictionary to store vehicle trajectories
trajectories = {}
first_coordinates = {}
last_coordinates = {}
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

        # Update the trajectory for this vehicle
        if id in trajectories:
            trajectories[id].append((cx, cy))
        else:
            trajectories[id] = [(cx, cy)]
            # Save first coordinates for this vehicle
            first_coordinates[id] = (cx, cy)

    # Draw trajectory lines on the frame
    draw_trajectory(img, trajectories)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save last coordinates for each vehicle
for vehicle_id, positions in trajectories.items():
    last_coordinates[vehicle_id] = positions[-1]

# Print first and last coordinates for each vehicle
print("First coordinates:")
print(first_coordinates)
print("Last coordinates:")
print(last_coordinates)

