import cv2
import numpy as np
import csv
import os
from glob import glob

def extract_extended_features(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    ratio_s_h = mean_s / (mean_h + 1e-5)
    ratio_v_s = mean_v / (mean_s + 1e-5)

    hist_h = cv2.calcHist([hsv], [0], None, [180], [0,180])
    hist_h = hist_h / hist_h.sum()
    entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))

    total_pixels = h.size
    prop_kuning = np.sum((h >= 20) & (h <= 40)) / total_pixels
    prop_hijau  = np.sum((h >= 50) & (h <= 70)) / total_pixels

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_gray = np.std(gray)

    return np.array([
        mean_h, mean_s, mean_v,
        std_h, std_s, std_v,
        ratio_s_h, ratio_v_s,
        entropy_h,
        prop_kuning, prop_hijau,
        std_gray
    ])

feature_names = [
    "Mean H", "Mean S", "Mean V",
    "Std H", "Std S", "Std V",
    "Ratio S/H", "Ratio V/S",
    "Entropy H",
    "Prop Kuning", "Prop Hijau",
    "Std Gray",
    "Label", "Filename"
]

csv_filename = 'fitur_buah.csv'
img_folder = 'gambar'
os.makedirs(img_folder, exist_ok=True)

# Header CSV
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    crop_w = int(w * 0.25)
    crop_h = int(h * 0.25)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    roi = frame[y1:y2, x1:x2]
    features = extract_extended_features(roi)

    # Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(overlay, "Center fruit here!", (x1 - 30, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    for idx, value in enumerate(features):
        text = f"{feature_names[idx]}: {value:.2f}"
        cv2.putText(overlay, text, (10, 30 + idx*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(overlay, "Press [m]=Matang  [r]=Mentah  [b]=Busuk", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Webcam + Fitur + Simpan CSV", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('m'), ord('r'), ord('b')]:
        label = {'m': 'matang', 'r': 'mentah', 'b': 'busuk'}[chr(key)]

        # Tentukan nomor urut berikutnya
        existing = glob(os.path.join(img_folder, f"{label}-*.jpg"))
        next_num = len(existing) + 1
        img_filename = f"{label}-{next_num}.jpg"
        img_path = os.path.join(img_folder, img_filename)

        # Simpan gambar ROI dan data fitur
        cv2.imwrite(img_path, roi)
        data_row = list(features) + [label, img_filename]
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
        print(f"Data disimpan: {img_filename} (label: {label})")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
