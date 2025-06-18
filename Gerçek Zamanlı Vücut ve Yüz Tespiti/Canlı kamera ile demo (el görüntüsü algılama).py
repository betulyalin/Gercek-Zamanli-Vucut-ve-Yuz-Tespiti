import cv2
import mediapipe as mp

# MediaPipe Hands ve çizim modülleri
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Kamera aç
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü BGR'den RGB'ye çevir
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # El tespiti yap
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Eğer el varsa
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # El noktalarının x,y koordinatlarını piksel cinsinden hesapla
                h, w, c = image.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

                # En küçük ve en büyük koordinatlarla dikdörtgen oluştur
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Dikdörtgen çiz (yeşil renk, kalınlık 2)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow('Canlı El Algılama', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
