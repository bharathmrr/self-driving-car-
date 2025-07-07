import cv2
import numpy as np

def get_distance(box_height, known_height=1.6, focal_length=615):
    return (known_height * focal_length) / box_height

cfg = r"C:\Users\lenovo\Pictures\yolov4-tiny.cfg"
wtg = r"C:\Users\lenovo\Downloads\yolov4-tiny.weights"
file = r"C:\Users\lenovo\Pictures\coco.names.txt"

net = cv2.dnn.readNet(wtg, cfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
with open(file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

video = cv2.VideoCapture(r"C:\Users\lenovo\Downloads\3055765-uhd_3840_2160_24fps.mp4")
frame_id = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:
        continue

    frame = cv2.resize(frame, (640, 360))
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                distance = get_distance(h)
                label = f"{classes[class_id]} {int(distance)}m"

                color = (0, 0, 255) if distance < 8 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if distance < 6:
                    cv2.putText(frame, "WARNING: Vehicle Too Close!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Collision Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
