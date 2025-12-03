# --- Импорт необходимых библиотек ---
import cv2
from ultralytics import YOLO

print("Загрузка модели YOLOv8-Pose...")
model = YOLO('yolov8n-pose.pt')
print("Модель успешно загружена.")

video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видеофайл по пути: {video_path}")
    exit()

# --- НОВОЕ ИЗМЕНЕНИЕ 1: Создаем окно заранее ---
# Мы создаем окно с флагом cv2.WINDOW_NORMAL, что делает его
# размер изменяемым с помощью мыши.
window_name = "Namaz Guide AI - Demonstration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

frame_number = 0
print("Начало обработки видео...")
while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_number += 1
        results = model(frame, verbose=False)

        if results[0].keypoints and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            print(f"Кадр {frame_number}: Обнаружен скелет из {len(keypoints)} точек.")
        else:
            print(f"Кадр {frame_number}: Скелет не обнаружен.")

        annotated_frame = results[0].plot()

        # --- НОВОЕ ИЗМЕНЕНИЕ 2: Масштабируем кадр для отображения ---
        # Чтобы огромное вертикальное видео поместилось на экране,
        # мы уменьшим его высоту до 800 пикселей, сохранив пропорции.
        # Вы можете изменить 800 на другое значение, если хотите.
        desired_height = 800
        original_height, original_width, _ = annotated_frame.shape
        
        # Рассчитываем коэффициент масштабирования
        scale_factor = desired_height / original_height
        
        # Новые размеры
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Изменяем размер кадра
        resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
        
        # --- Отображаем ИЗМЕНЕННЫЙ кадр в НАШЕМ окне ---
        cv2.imshow(window_name, resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Выход по нажатию клавиши 'q'.")
            break
    else:
        print("Видеофайл обработан до конца.")
        break

cap.release()
cv2.destroyAllWindows()
print("Все ресурсы освобождены. Программа завершена.")