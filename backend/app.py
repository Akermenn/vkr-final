import matplotlib

matplotlib.use('Agg')  # Обязательно: без экрана
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score

app = Flask(__name__)

# Папка для картинок
IMG_FOLDER = os.path.join(os.getcwd(), 'static')
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# Глобальная переменная для результатов
results_data = {}


def run_analysis():
    print("--- STARTING FULL MODEL TRAINING ---")
    try:
        # 0. Загрузка данных
        df = pd.read_csv('ore_classification_dataset.csv')
        if 'Sample_ID' in df.columns: df = df.drop('Sample_ID', axis=1)

        # Добавляем синтетические признаки (как было у тебя)
        df['Si_Fe_Ratio'] = df['SiO2_pct'] / (df['Fe2O3_pct'] + 0.001)
        df['Al_Mg_Ratio'] = df['Al2O3_pct'] / (df['MgO_pct'] + 0.001)

        X = df.drop('Ore_Type', axis=1)
        y = df['Ore_Type']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Препроцессинг
        cat_cols = ['Lithology', 'Mineralogy']
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # --- 1. КОРРЕЛЯЦИЯ ---
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_FOLDER, 'correlation.png'))
        plt.close()

        # --- 2. КЛАССИФИКАЦИЯ (Все модели) ---
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
        }

        class_results = {}
        for name, model in models.items():
            try:
                clf = Pipeline(steps=[('prep', preprocessor), ('model', model)])
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test))
                class_results[name] = round(acc, 4)
            except Exception as e:
                print(f"Error in {name}: {e}")
                class_results[name] = "Error"

        results_data['classification'] = class_results

        # --- 3. ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest) ---
        rf_pipe = Pipeline(
            steps=[('prep', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
        rf_pipe.fit(X_train, y_train)
        try:
            ohe_features = rf_pipe.named_steps['prep'].transformers_[1][1].get_feature_names_out(cat_cols)
            feats = num_cols + list(ohe_features)
            imps = rf_pipe.named_steps['model'].feature_importances_
            feat_imp = pd.DataFrame({'Feature': feats, 'Importance': imps}).sort_values('Importance',
                                                                                        ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
            plt.tight_layout()
            plt.savefig(os.path.join(IMG_FOLDER, 'importance.png'))
            plt.close()
        except:
            pass

        # --- 4. РЕГРЕССИЯ ---
        X_reg = df.drop(['Ore_Type', 'SiO2_pct', 'Si_Fe_Ratio'], axis=1)
        y_reg = df['SiO2_pct']

        num_cols_reg = X_reg.select_dtypes(include=['int64', 'float64']).columns.tolist()
        preprocessor_reg = ColumnTransformer([
            ('num', StandardScaler(), num_cols_reg),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        reg_pipe = Pipeline(
            [('prep', preprocessor_reg), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
        reg_pipe.fit(X_train_r, y_train_r)
        y_pred_r = reg_pipe.predict(X_test_r)
        r2 = r2_score(y_test_r, y_pred_r)
        results_data['regression_r2'] = round(r2, 4)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test_r, y_pred_r, alpha=0.5, color='blue')
        plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
        plt.title('Regression: True vs Predicted SiO2')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_FOLDER, 'regression.png'))
        plt.close()

        # --- 5. КЛАСТЕРИЗАЦИЯ (K-Means + PCA) ---
        X_processed = preprocessor.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_processed)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', s=50)
        plt.title('K-Means Clustering (PCA Projection)')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_FOLDER, 'clustering.png'))
        plt.close()

        print("--- TRAINING FINISHED SUCCESS ---")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        results_data['error'] = str(e)


# Запускаем
run_analysis()


@app.route('/')
def index(): return "Backend OK"


@app.route('/api/data')
def get_data():
    response = jsonify(results_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/static/<path:filename>')
def serve_image(filename):
    response = send_from_directory(IMG_FOLDER, filename)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)