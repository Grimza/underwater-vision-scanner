from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

img = cv2.imread("assets/demo_img_1.webp")

results = model(img)

results[0].show()