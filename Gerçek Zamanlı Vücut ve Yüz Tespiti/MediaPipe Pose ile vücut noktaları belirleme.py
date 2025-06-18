import cv2
import numpy as np
import requests
import mediapipe as mp

# Görsel URL'si
url = 'https://images.pexels.com/photos/775358/pexels-photo-775358.jpeg'

# Görseli internetten oku
response = requests.get(url)
img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# MediaPipe Pose yükle ve çiz
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=True) as pose:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
        )
        print("Vücut noktaları başarıyla çizildi.")
    else:
        print("Hiçbir vücut noktası tespit edilemedi.")

cv2.namedWindow("MediaPipe Pose ile İskelet Çizimi", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MediaPipe Pose ile İskelet Çizimi", 800, 600)  # Başlangıç boyutu

while True:
    # Pencerenin mevcut boyutunu al
    x, y, w, h = cv2.getWindowImageRect("MediaPipe Pose ile İskelet Çizimi")

    # Görüntüyü pencere boyutuna orantılı ölçekle
    scale = min(w / image.shape[1], h / image.shape[0])
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Siyah arka planlı boş resim oluştur (pencere boyutunda)
    background = np.zeros((h, w, 3), dtype=np.uint8)

    # Resmi ortala
    x_offset = (w - new_w) // 2
    y_offset = (h - new_h) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    cv2.imshow("MediaPipe Pose ile İskelet Çizimi", background)

    key = cv2.waitKey(100)
    if key == 27 or key == ord('q'):  # ESC veya q ile çık
        break

cv2.destroyAllWindows()
