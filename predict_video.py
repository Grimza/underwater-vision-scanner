from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

cap = cv2.VideoCapture("assets/your_video.webm")

# cap.set(cv2.CAP_PROP_POS_MSEC, 60000 * 1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

frame_interval = int(fps * 2) # 2 seconds

frame_count = 0
last_boxes = None  # only boxes

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = model(frame)
        last_boxes = results[0].boxes

    display = frame.copy()

    if last_boxes is not None:
        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            if conf > 0.7: # how much is similar to the tags... 
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cls = int(box.cls)
                label = model.names[cls]

                text = f"{label} {conf:.2f}"

                cv2.putText(
                    display,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    display = cv2.resize(display, (1280, 720))
    cv2.imshow("Detection", display)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()