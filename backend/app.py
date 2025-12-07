import matplotlib

matplotlib.use('Agg')  # Важно: отключаем экран, чтобы сервер не упал
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from flask import Flask, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

app = Flask(__name__)

# Папка для картинок
IMG_FOLDER = os.path.join(os.getcwd(), 'static')
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# Глобальная переменная для результатов
results_data = {}


def run_analysis():
    print("--- STARTING MODEL TRAINING ---")
    try:
        df = pd.read_csv('ore_classification_dataset.csv')
        if 'Sample_ID' in df.columns: df = df.drop('Sample_ID', axis=1)

        X = df.drop('Ore_Type', axis=1)
        y = df['Ore_Type']

        # Препроцессинг
        cat_cols = ['Lithology', 'Mineralogy']
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. График корреляции
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_FOLDER, 'correlation.png'))
        plt.close()

        # 2. Обучение (Классификация)
        clf = Pipeline(
            steps=[('prep', preprocessor), ('model', RandomForestClassifier(n_estimators=50, random_state=42))])
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results_data['accuracy'] = round(acc, 4)

        # 3. Важность признаков
        try:
            ohe_features = clf.named_steps['prep'].transformers_[1][1].get_feature_names_out(cat_cols)
            feats = num_cols + list(ohe_features)
            imps = clf.named_steps['model'].feature_importances_
            feat_imp = pd.DataFrame({'Feature': feats, 'Importance': imps}).sort_values('Importance',
                                                                                        ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
            plt.tight_layout()
            plt.savefig(os.path.join(IMG_FOLDER, 'importance.png'))
            plt.close()
        except:
            pass

        print("--- TRAINING FINISHED ---")
    except Exception as e:
        print(f"Error during training: {e}")


# Запускаем обучение сразу при старте
run_analysis()


@app.route('/')
def index(): return "Backend OK"


@app.route('/api/data')
def get_data():
    response = jsonify(results_data)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Разрешаем фронтенду брать данные
    return response


@app.route('/static/<path:filename>')
def serve_image(filename):
    response = send_from_directory(IMG_FOLDER, filename)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)