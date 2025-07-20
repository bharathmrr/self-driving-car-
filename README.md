# 🚗 Self-Driving Car: Collision Detection & Direction Guidance

This project uses **YOLOv4-tiny** and **OpenCV** to simulate a basic **self-driving car perception system**, detecting nearby vehicles, estimating distance, and giving safety warnings and direction guidance (left, right, straight).

---

## 🎯 Features

- Real-time object detection using **YOLOv4-tiny**
- Calculates distance to detected cars using bounding box height
- Flags vehicles as **Safe** or **Danger**
- Predicts optimal movement direction:
  - 🔴 Danger ahead → Suggest left or right
  - ✅ Safe distance → Maintain direction

---

## 🧠 How It Works

1. Input video is processed frame-by-frame.
2. YOLOv4-tiny detects **cars** in the scene.
3. For each car, the distance is estimated using the formula:

