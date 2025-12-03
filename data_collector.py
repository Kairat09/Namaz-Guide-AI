import cv2
import csv
import os
from ultralytics import YOLO

# --- Настройка ---
MODEL = YOLO('yolov8n-pose.pt')
VIDEO_PATH = 'test_video.mp4' # Используем то же видео
OUTPUT_CSV_PATH = 'poses_data.csv'

# Определяем названия поз и соответствующие им клавиши
POSE_MAPPING = {
    's': 'standing',  # s - стоит
    'b': 'bowing',    # b - поклон (руку')
    'p': 'prostrating', # p - земной поклон (саджда)
    'j': 'sitting',   # j - сидит (джулюс)
    'x': 'other'      # x - пропустить кадр (переходное движение и т.д.)
}

# --- Подготовка CSV файла ---
# Создаем файл и записываем в него заголовок
def prepare_csv():
    # Нам нужно 17 точек, у каждой по 2 координаты (x, y). Итого 34 колонки + 1 для метки.
    header = ['pose_label']
    for i in range(17):
        header += [f'point_{i}_x', f'point_{i}_y']
    
    # Записываем заголовок в файл, только если файл еще не создан
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

# --- Основная функция ---
def collect_data():
    prepare_csv()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {VIDEO_PATH}")
        return

    window_name = "Data Collector - Press a key to label pose"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- Начинаем сбор данных ---")
    print("В активном окне с видео нажимайте клавиши для разметки:")
    for key, pose in POSE_MAPPING.items():
        print(f"  '{key}' -> {pose}")
    print("--------------------------------\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Получаем предсказания модели
        results = MODEL(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Отображаем кадр, уменьшенный для удобства
        desired_height = 800
        h, w, _ = annotated_frame.shape
        scale = desired_height / h
        resized_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        cv2.imshow(window_name, resized_frame)
        
        # Ждем нажатия клавиши
        key = cv2.waitKey(0) & 0xFF  # waitKey(0) ставит видео на паузу до нажатия

        # Выход из программы по клавише 'q'
        if key == ord('q'):
            print("Сбор данных прерван пользователем.")
            break
        
        # Конвертируем код клавиши в символ
        key_char = chr(key).lower()

        # Если нажатая клавиша есть в нашем словаре POSE_MAPPING
        if key_char in POSE_MAPPING:
            pose_name = POSE_MAPPING[key_char]

            # Если поза не 'other', сохраняем данные
            if pose_name != 'other':
                if results[0].keypoints and len(results[0].keypoints.xy) > 0:
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    
                    # Преобразуем массив 17x2 в плоский список из 34 элементов
                    flat_keypoints = keypoints.flatten().tolist()
                    
                    # Создаем строку для записи в CSV: [название_позы, x0, y0, x1, y1, ...]
                    csv_row = [pose_name] + flat_keypoints
                    
                    # Дописываем строку в конец файла
                    with open(OUTPUT_CSV_PATH, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_row)
                    
                    print(f"Сохранен кадр с позой: {pose_name}")
                else:
                    print("Пропущено: скелет не обнаружен на кадре.")
            else:
                print("Кадр пропущен (помечен как 'other').")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nСбор данных завершен. Результаты сохранены в {OUTPUT_CSV_PATH}")


# --- Запуск сбора данных ---
if __name__ == "__main__":
    collect_data()