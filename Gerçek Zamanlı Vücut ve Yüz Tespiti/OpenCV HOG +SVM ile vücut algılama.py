import cv2
import requests
import numpy as np

# Resim URL'si
url = 'https://images.pexels.com/photos/26732883/pexels-photo-26732883.jpeg'

# Resmi URL'den oku
response = requests.get(url)
img_array = np.array(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    print("Resim yüklenemedi! URL'yi kontrol edin.")
    exit()

# Ekran çözünürlüğü (örnek: 1920x1080)
screen_width, screen_height = 1920, 1080

# Görüntü oranlarını hesapla
img_h, img_w = img.shape[:2]
scale_w = screen_width / img_w
scale_h = screen_height / img_h
scale = min(scale_w, scale_h)

# Görüntüyü uygun ölçekte yeniden boyutlandır
new_w = int(img_w * scale)
new_h = int(img_h * scale)
resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Siyah arka plan oluştur
canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# Ortalamak için offset hesapla
x_offset = (screen_width - new_w) // 2
y_offset = (screen_height - new_h) // 2

# Resmi siyah arka planın ortasına yerleştir
canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

# HOG tanımlayıcı oluştur ve insan dedektörü ayarla
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# İnsanları algıla
(rects, weights) = hog.detectMultiScale(canvas, winStride=(8,8), padding=(16,16), scale=1.05)

# Algılanan kişilerin çevresine dikdörtgen çiz
for (x, y, w, h) in rects:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 3)

print(f"{len(rects)} kişi bulundu.")

# Tam ekran pencere oluştur ve göster
cv2.namedWindow("HOG + SVM ile Vücut Algılama", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("HOG + SVM ile Vücut Algılama", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("HOG + SVM ile Vücut Algılama", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
