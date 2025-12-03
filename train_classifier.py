import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle # Библиотека для сохранения модели

# --- 1. Загрузка данных ---
CSV_FILE_PATH = 'poses_data.csv'

# Используем pandas для чтения CSV файла
print(f"Загрузка данных из {CSV_FILE_PATH}...")
df = pd.read_csv(CSV_FILE_PATH)

# Проверяем, что данные загрузились
if df.empty:
    print("Ошибка: CSV файл пуст. Пожалуйста, соберите данные с помощью data_collector.py")
    exit()

print(f"Загружено {len(df)} строк данных.")

# --- 2. Подготовка данных ---
# Разделяем данные на признаки (X) и целевую переменную (y)
# X - это все колонки с координатами (все, кроме первой)
# y - это первая колонка с названием позы (pose_label)
X = df.drop('pose_label', axis=1) # axis=1 означает, что мы удаляем колонку
y = df['pose_label']

# --- 3. Разделение на обучающую и тестовую выборки ---
# Мы разделим наши данные: 80% на обучение, 20% на тестирование.
# Это нужно, чтобы проверить, насколько хорошо модель работает на данных,
# которые она никогда не видела во время обучения.
# random_state=42 гарантирует, что разделение будет всегда одинаковым.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Данные разделены: {len(X_train)} для обучения, {len(X_test)} для теста.")

# --- НОВОЕ ИЗМЕНЕНИЕ: Конвертируем в NumPy, чтобы убрать названия колонок ---
X_train = X_train.values
X_test = X_test.values

# --- 4. Обучение модели ---
# Мы будем использовать простую, но эффективную модель - Логистическую регрессию.
# max_iter=1000 - увеличиваем количество итераций для лучшей сходимости.
model = LogisticRegression(max_iter=1000)

print("Начинаем обучение модели...")
# Это и есть процесс обучения. Модель анализирует X_train и y_train.
model.fit(X_train, y_train)
print("Модель успешно обучена.")

# --- 5. Оценка качества модели ---
# Делаем предсказания на тестовых данных
y_pred = model.predict(X_test)

# Сравниваем предсказания (y_pred) с реальными метками (y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовых данных: {accuracy * 100:.2f}%")

# --- 6. Сохранение модели ---
# Теперь, когда модель обучена, мы сохраняем ее в файл,
# чтобы не обучать ее каждый раз заново.
MODEL_SAVE_PATH = 'pose_classifier.pkl'
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Модель успешно сохранена в файл: {MODEL_SAVE_PATH}")