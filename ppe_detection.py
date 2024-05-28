from ultralytics import YOLO
import cv2
import cvzone
import math

#### For Webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

### For Video
cap = cv2.VideoCapture("videos/ppe-3.mp4")

model = YOLO("model/best.pt")

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV',
              'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer',
              'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

# Init myColor
myColor = (0, 0, 255) # red

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Height and width
            w, h = x2-x1, y2-y1            

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100 # or conf = f"{box.conf[0]:.2f}"
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            myColor = (0, 0, 255)

            # Colours based on class
            if conf>0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == 'NO-Mask':
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass =='Safety Vest' or currentClass == 'Mask':
                    myColor =(0, 255, 0)
                else:
                    myColor = (255, 0, 0)

            # Add texts in rectangle
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                colorT=(255,255,255),colorR=myColor, offset=5)
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)            
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)

