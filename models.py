from ultralytics import YOLO
import cv2

model = YOLO("trainedmodels/phoneusage.pt")
img = cv2.imread("image.png")  # take ANY frame from video

results = model(img, conf=0.1)

print(results[0].boxes)
