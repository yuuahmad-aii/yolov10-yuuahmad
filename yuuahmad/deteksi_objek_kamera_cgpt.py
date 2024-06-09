import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOv10

# Inisialisasi variabel
count = 0
detect_area = ((100, 200), (500, 300))  # Area deteksi: ((x1, y1), (x2, y2))

# Inisialisasi background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Membuka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop frame ke area deteksi
    detect_region = frame[detect_area[0][1]:detect_area[1]
                          [1], detect_area[0][0]:detect_area[1][0]]

    # Terapkan background subtractor
    fgmask = fgbg.apply(detect_region)

    # Menentukan kontur
    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Hanya hitung objek dengan area lebih dari 500 px
        if cv2.contourArea(contour) > 500:
            count += 1
            break  # Hanya hitung sekali per frame

    # Menampilkan jumlah objek yang terdeteksi
    cv2.putText(frame, f'Count: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Menampilkan area deteksi pada frame
    cv2.rectangle(frame, detect_area[0], detect_area[1], (255, 0, 0), 2)

    # Menampilkan frame
    cv2.imshow('Frame', frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Membersihkan
cap.release()
cv2.destroyAllWindows()
