import cv2
import os
import csv
from ultralytics import YOLO

model_path = 'C:/Users/Rama Eka/Documents/Tugas/semester 7/comvis/test/skripsi/training/yolov5/runs/detect/Flare Detection (3)/weights/best.pt'
input_folder = 'H:/Download/Compressed/Flare Dataset/Testing/Good'  
output_folder = os.path.join(input_folder, 'detections')  

# Buat folder output
os.makedirs(output_folder, exist_ok=True)

model = YOLO(model_path)

# Dictionary untuk menghitung jumlah deteksi per kelas
class_counts = {}

# Loop semua file gambar di folder input
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Gagal membaca {filename}")
            continue

        # Deteksi objek
        results = model.predict(image, conf=0.3)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                # Hitung jumlah deteksi per kelas
                class_counts[label] = class_counts.get(label, 0) + 1

                # Tulis label dan confidence dibounding box
                label_text = f'{label} {conf:.2f}'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        # Simpen gambat
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        print(f"Hasil deteksi disimpan ke: {output_path}")

# simpen ke csv
csv_path = os.path.join(output_folder, 'class_counts.csv')
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Count'])
    for cls, count in class_counts.items():
        writer.writerow([cls, count])

