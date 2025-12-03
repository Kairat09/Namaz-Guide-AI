import cv2
import pickle
import numpy as np
from ultralytics import YOLO

# --- 1. Загрузка моделей ---
pose_model = YOLO('yolov8n-pose.pt')

CLASSIFIER_MODEL_PATH = 'pose_classifier.pkl'
with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
    pose_classifier = pickle.load(f)

print("Все модели успешно загружены.")

# --- 2. Настройка видео ---
VIDEO_PATH = 'test_video.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видео {VIDEO_PATH}")
    exit()

window_name = "Namaz Guide AI - Live Prediction"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- НОВОЕ: Инициализация переменных для логики подсчета ---
rakah_counter = 0
sajda_counter = 0
current_pose_state = None  # Хранит текущее состояние позы

print("Начинаем предсказание поз и подсчет ракаатов...")

# --- 3. Основной цикл ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = pose_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        flat_keypoints = keypoints.flatten()
        input_data = np.array([flat_keypoints])
        
        predicted_pose = pose_classifier.predict(input_data)[0]
        prediction_probability = pose_classifier.predict_proba(input_data)[0]
        confidence = max(prediction_probability)
        
        # --- НОВОЕ: ЛОГИКА КОНЕЧНОГО АВТОМАТА (STATE MACHINE) ---
        
        # Проверяем, изменилась ли поза с предыдущего кадра
        if predicted_pose != current_pose_state:
            
            # Мы засчитываем саджда только при ПЕРЕХОДЕ в это состояние
            if predicted_pose == 'prostrating':
                sajda_counter += 1
                
                # Проверяем, завершен ли ракаат (после второго саджда)
                if sajda_counter == 2:
                    rakah_counter += 1
                    sajda_counter = 0 # Сбрасываем счетчик для следующего ракаата
            
            # Обновляем текущее состояние
            current_pose_state = predicted_pose

        # --- Отображаем всю информацию на кадре ---
        
        # Настройки для текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # Отображаем текущую позу
        pose_text = f"Pose: {predicted_pose} ({confidence:.2f})"
        cv2.putText(annotated_frame, pose_text, (50, 50), font, font_scale, (0, 255, 0), thickness)
        
        # НОВОЕ: Отображаем счетчики
        sajda_text = f"Sajda Count: {sajda_counter}"
        cv2.putText(annotated_frame, sajda_text, (50, 110), font, font_scale, (255, 255, 0), thickness)
        
        rakah_text = f"Rak'ah Count: {rakah_counter}"
        cv2.putText(annotated_frame, rakah_text, (50, 170), font, font_scale, (0, 165, 255), thickness)
        
    # --- Отображение кадра ---
    desired_height = 800
    h, w, _ = annotated_frame.shape
    scale = desired_height / h
    resized_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Работа завершена.")