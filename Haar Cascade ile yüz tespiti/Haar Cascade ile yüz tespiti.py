#Bu kodu aktif kullanabilmek için aşağıdaki GitHub üzerinden alınan ve urlsi verilen dosyası bilgiayarınıza indirmeniz gereklidir.
import cv2
import numpy as np
import urllib.request
import os

# 1. Cascade dosyasını indir (bir kez)
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    url = 'https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, cascade_path)
    print(f"{cascade_path} dosyası indirildi.")

# 2. Cascade yükle
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Cascade dosyası yüklenemedi!")
    exit()

# 3. Resmi URL'den indir (bu sefer gerçek insan fotoğrafı)
img_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
resp = urllib.request.urlopen(img_url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(image, cv2.IMREAD_COLOR)

# 4. Yüz algılama
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

if len(faces) == 0:
    print("Yüz bulunamadı.")
else:
    print(f"{len(faces)} yüz bulundu.")
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 5. Sonucu göster
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

