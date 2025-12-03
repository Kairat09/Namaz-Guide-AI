import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pickle
import numpy as np

# --- Кэширование моделей ---
# Загружаем YOLO один раз и храним в кэше
@st.cache_resource
def load_yolo_model():
    model = YOLO('yolov8n-pose.pt')
    return model

# Загружаем наш классификатор один раз и храним в кэше
@st.cache_resource
def load_classifier_model():
    with open('pose_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# --- Основная часть приложения ---

st.set_page_config(layout="wide", page_title="Namaz Guide AI")

st.title("Namaz Guide AI 🤖")
st.write("---")
st.subheader("Ваш персональный ассистент для анализа намаза")
st.write("""
Этот проект использует компьютерное зрение для распознавания поз в намазе в реальном времени
и автоматического подсчета ракаатов. Загрузите видеофайл для начала анализа.
""")

# Загружаем модели с помощью кэшированных функций
pose_model = load_yolo_model()
pose_classifier = load_classifier_model()

# Создаем виджет для загрузки файла
uploaded_file = st.file_uploader("Выберите видеофайл (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Сохраняем загруженный файл во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # --- Инициализация переменных для логики ---
    rakah_counter = 0
    sajda_counter = 0
    current_pose_state = None

    st.success(f"Видео успешно загружено. Начинаем обработку...")
    
    cap = cv2.VideoCapture(video_path)
    
    # --- Создаем "места" на странице для вывода ---
    col1, col2 = st.columns([2, 1]) # Видео будет занимать 2/3, метрики 1/3
    with col1:
        frame_placeholder = st.empty()
    with col2:
        st.subheader("Показатели анализа:")
        pose_placeholder = st.empty()
        sajda_placeholder = st.empty()
        rakah_placeholder = st.empty()

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
            
            # --- Логика конечного автомата (State Machine) ---
            if predicted_pose != current_pose_state:
                if predicted_pose == 'prostrating':
                    sajda_counter += 1
                    if sajda_counter == 2:
                        rakah_counter += 1
                        sajda_counter = 0
                current_pose_state = predicted_pose

        # Отображаем видеокадр
        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Обновляем метрики на странице
        pose_placeholder.metric("Текущая поза", current_pose_state if current_pose_state else "Определение...")
        sajda_placeholder.metric("Счетчик саджда", sajda_counter)
        rakah_placeholder.metric("Счетчик ракаатов", rakah_counter)
        
    st.info("Обработка видео завершена.")
    cap.release()