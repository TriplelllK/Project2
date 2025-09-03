import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

def train_and_save_model():
    file_path = '/mnt/c/Users/Kuat/Documents/AI engineering course/.venv/HW2/data/bank_full.csv'
    if not os.path.exists(file_path):
        print(f"Ошибка: файл '{file_path}' не найден.")
        return

    try:
        data = pd.read_csv(file_path, sep=';')
        if data.shape[1] <= 2 and ';' not in data.columns[0]:
            raise ValueError("Separator might be ';'. Trying ','.")
    except (pd.errors.ParserError, ValueError):
        try:
            data = pd.read_csv(file_path, sep=',')
            print("Файл загружен с разделителем ','.")
        except Exception as e:
            print(f"Ошибка при загрузке файла CSV с разделителем ',': {e}")
            return
    except Exception as e:
        print(f"Ошибка при загрузке файла CSV: {e}")
        return

    if 'y' not in data.columns:
        print("Ошибка: Колонка 'y' не найдена в датасете. Пожалуйста, убедитесь, что целевая переменная названа 'y'.")
        print(f"Доступные колонки: {data.columns.tolist()}")
        return

    # Выбираем 10 заданных признаков
    selected_features = [
        'poutcome', 'contact', 'duration', 'housing', 'month',
        'previous', 'pdays', 'loan', 'age', 'day'
    ]

    # Проверка на наличие выбранных признаков в датасете
    missing_selected_features = [f for f in selected_features if f not in data.columns]
    if missing_selected_features:
        print(f"Ошибка: Признаки отсутствуют в датасете: {missing_selected_features}")
        print(f"Доступные колонки: {data.columns.tolist()}")
        return

    X = data[selected_features].copy() # Обучение только с этими признаками
    y = data['y']

    # Кодируем целевую переменную 'y'
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)
    print(f"Целевая переменная 'y' закодирована: {y_encoder.classes_.tolist()} -> {list(range(len(y_encoder.classes_)))}")

    encoders = {}
    features_info = {}
    fallback_values = {}

    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            temp_col_for_mode = X[column].fillna(X[column].mode()[0])
            mode_value = temp_col_for_mode.mode()[0]

            le.fit(X[column].astype(str).unique())
            
            features_info[column] = {
                'type': 'categorical',
                'values': le.classes_.tolist(),
                'mode': mode_value
            }
            encoders[column] = le
            fallback_values[column] = mode_value

        else:
            features_info[column] = {'type': 'numerical', 'min': X[column].min(), 'max': X[column].max()}
            fallback_values[column] = X[column].median()

    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = X[column].fillna(fallback_values[column])
            X[column] = encoders[column].transform(X[column])
        else:
            X[column] = X[column].fillna(fallback_values[column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели XGBoost. Данная модель выбрана так как она хорошо подходит для задач классификации и регрессии. 
    # По опыту предыдущего домашнего задания, она показала хорошие результаты на этом датасете.
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Определение и вывод топ-10 важных признаков. Будет отображено 10 признаков, которыми модель обучилась.
    # Признаки были отобраны исходя из опыта предыдущего домашнего задания.
    top_10_features_names = selected_features
    print("\nМодель обучена только на следующих 10 признаках:")
    print(top_10_features_names)
    

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Создана директория: '{models_dir}'")

    # Сохранение параметров XGBoost в TXT-файл
    try:
        xgboost_params = model.get_params()
        params_string = ""
        for key, value in xgboost_params.items():
            # Преобразуем типы numpy в стандартные типы Python
            if isinstance(value, (np.integer, np.floating)):
                params_string += f"{key}: {value.item()}\n"
            else:
                params_string += f"{key}: {value}\n"

        params_filename = os.path.join(models_dir, 'xgboost_params.txt')
        with open(params_filename, 'w') as f:
            f.write(params_string)
        print(f"Параметры XGBoost успешно сохранены в '{params_filename}'")
    except Exception as e:
        print(f"Ошибка при сохранении параметров XGBoost в TXT: {e}")
    

    # Создаем словарь, содержащий все артефакты
    all_artifacts = {
        'model': model,
        'features_list': X.columns.tolist(), # Список всех признаков, на которых обучалась модель
        'label_encoders': encoders,
        'features_info': features_info,
        'y_encoder': y_encoder,
        'fallback_values': fallback_values,
        'top_10_features_names': top_10_features_names
    }

    # Сохраняем весь словарь в один файл .joblib
    artifacts_filename = os.path.join(models_dir, 'all_bank_marketing_artifacts.joblib')
    joblib.dump(all_artifacts, artifacts_filename)
    print(f"\nВсе артефакты успешно обучены и сохранены в '{artifacts_filename}'")

if __name__ == '__main__':
    train_and_save_model()
