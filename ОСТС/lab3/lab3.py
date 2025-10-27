# ============================================================================
# ПОВНИЙ PIPELINE: ЗАВАНТАЖЕННЯ, АНАЛІЗ, ОПТИМІЗАЦІЯ, ЗБЕРЕЖЕННЯ
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import pickle
import os
import re
import ast
import warnings
from datetime import datetime
from collections import Counter

# SQLAlchemy та PostgreSQL
from sqlalchemy import create_engine, text
import psycopg2

# Scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score,
    classification_report, confusion_matrix
)

# XGBoost
from xgboost import XGBClassifier

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Scipy для гіперпараметрів
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

# ============================================================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================================
print("="*70)
print("📂 ЗАВАНТАЖЕННЯ ДАНИХ")
print("="*70)

file_path = "D:/5sem/ОСТС/lab3/customer_booking.lab2.2.csv"

# Визначення кодування
with open(file_path, "rb") as f:
    result = chardet.detect(f.read(50000))
print(f"Визначено кодування: {result['encoding']}")

# Завантаження CSV
dataset = pd.read_csv(file_path, encoding='Windows-1252')
print(f"✅ Дані завантажено: {dataset.shape}")
print(f"   Рядків: {dataset.shape[0]}, Колонок: {dataset.shape[1]}")

# Базова інформація
print("\n📊 Інформація про датасет:")
dataset.info()

# ============================================================================
# 2. ПІДКЛЮЧЕННЯ ДО POSTGRESQL
# ============================================================================
print("\n" + "="*70)
print("🔗 ПІДКЛЮЧЕННЯ ДО POSTGRESQL")
print("="*70)

# Параметри підключення
user = "postgres"
password = "postgres"
host = "localhost"
port = "5432"
database_name = "customer_booking"

# Підключення до postgres (для створення БД)
engine_postgres = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres"
)

# Безпечне створення БД
try:
    with engine_postgres.connect() as conn:
        # Перевірка існування
        result = conn.execute(text(
            f"SELECT 1 FROM pg_database WHERE datname = '{database_name}'"
        ))
        exists = result.fetchone() is not None
        
        if not exists:
            conn.execute(text("COMMIT"))
            conn.execute(text(f"CREATE DATABASE {database_name}"))
            print(f"✅ База даних '{database_name}' створена.")
        else:
            print(f"ℹ️  База даних '{database_name}' вже існує.")
except Exception as e:
    print(f"⚠️  Помилка створення БД: {e}")

# Підключення до цільової БД
engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database_name}"
)

# Завантаження даних в таблицю
try:
    dataset.to_sql("flight_bookings", engine, if_exists='replace', index=False)
    print(f"✅ Дані завантажено в таблицю 'flight_bookings'")
except Exception as e:
    print(f"❌ Помилка завантаження даних: {e}")

# ============================================================================
# 3. СТВОРЕННЯ ТАБЛИЦЬ ДЛЯ МЕТРИК
# ============================================================================
print("\n" + "="*70)
print("🗄️  СТВОРЕННЯ ТАБЛИЦЬ ДЛЯ МЕТРИК")
print("="*70)

tables_sql = {
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100),
            source VARCHAR(50),
            accuracy FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "model_metrics": """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100),
            dataset_type VARCHAR(20),
            accuracy FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            precision FLOAT,
            recall FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "hyperparameter_optimization": """
        CREATE TABLE IF NOT EXISTS hyperparameter_optimization (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100),
            optimization_stage VARCHAR(20),
            accuracy FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            precision FLOAT,
            recall FLOAT,
            best_params TEXT,
            cv_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "saved_models": """
        CREATE TABLE IF NOT EXISTS saved_models (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100),
            filename VARCHAR(255),
            file_size_kb FLOAT,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "class_balancing_results": """
        CREATE TABLE IF NOT EXISTS class_balancing_results (
            "Method" VARCHAR(100),
            "Train_Size" INTEGER,
            "Accuracy" FLOAT,
            "F1_Score" FLOAT,
            "ROC_AUC" FLOAT,
            "Precision" FLOAT,
            "Recall" FLOAT
        );
    """
}

with engine.connect() as conn:
    for table_name, sql in tables_sql.items():
        conn.execute(text(sql))
        print(f"   ✅ {table_name}")
    conn.commit()

print("✅ Всі таблиці створено!")

# ============================================================================
# 4. ПІДГОТОВКА ДАНИХ
# ============================================================================
print("\n" + "="*70)
print("🔧 ПІДГОТОВКА ДАНИХ")
print("="*70)

# Розділення на features та target
target = 'booking_complete'
X = dataset.drop(columns=[target])
y = dataset[target]

print(f"\n📊 Баланс класів:")
print(y.value_counts(normalize=True))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Розділення даних:")
print(f"   Train: {X_train.shape[0]} зразків")
print(f"   Test:  {X_test.shape[0]} зразків")

# Стандартизація (для Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()

# ============================================================================
# 5. ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*70)
print("🤖 ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ")
print("="*70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

print(f"✅ Ініціалізовано {len(models)} моделей:")
for name in models.keys():
    print(f"   • {name}")

# ============================================================================
# 6. БАЗОВЕ НАВЧАННЯ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*70)
print("🎓 БАЗОВЕ НАВЧАННЯ МОДЕЛЕЙ")
print("="*70)

initial_results = []

for name, model in models.items():
    print(f"\n⏳ Навчання {name}...")
    
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    
    print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    initial_results.append({
        'model_name': name,
        'source': 'initial_train',
        'accuracy': float(acc),
        'f1_score': float(f1),
        'roc_auc': float(auc)
    })

# Збереження в БД
pd.DataFrame(initial_results).to_sql('predictions', engine, if_exists='append', index=False)
print("\n✅ Результати збережено в БД")

# ============================================================================
# 7. ВИБІР МЕТРИКИ ДЛЯ ОПТИМІЗАЦІЇ
# ============================================================================
print("\n" + "="*70)
print("🎯 ВИБІР МЕТРИКИ ДЛЯ ОПТИМІЗАЦІЇ")
print("="*70)

class_balance = y.value_counts(normalize=True)
imbalance_ratio = class_balance.min() / class_balance.max()

print(f"Коефіцієнт дисбалансу: {imbalance_ratio:.3f}")

if imbalance_ratio < 0.7:
    target_metric = 'f1'
    print(f"✅ Обрана метрика: F1-Score (класи незбалансовані)")
else:
    target_metric = 'roc_auc'
    print(f"✅ Обрана метрика: ROC-AUC (класи збалансовані)")

# ============================================================================
# 8. НАЛАШТУВАННЯ ГІПЕРПАРАМЕТРІВ
# ============================================================================
print("\n" + "="*70)
print("⚙️  НАЛАШТУВАННЯ ГІПЕРПАРАМЕТРІВ")
print("="*70)

param_distributions = {
    "Logistic Regression": {
        'C': uniform(0.01, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None]
    },
    "Random Forest": {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False]
    },
    "XGBoost": {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
}

print("✅ Параметри налаштовано для всіх моделей")

# ============================================================================
# 9. ГІПЕРПАРАМЕТРИЧНА ОПТИМІЗАЦІЯ
# ============================================================================
print("\n" + "="*70)
print("🚀 ЗАПУСК ГІПЕРПАРАМЕТРИЧНОЇ ОПТИМІЗАЦІЇ")
print("="*70)

os.makedirs('models', exist_ok=True)
optimization_results = []

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"🔍 Оптимізація: {model_name}")
    print(f"{'='*70}")
    
    # Метрики ДО оптимізації
    if model_name == "Logistic Regression":
        y_pred_before = model.predict(X_test_scaled)
    else:
        y_pred_before = model.predict(X_test)
    
    acc_before = accuracy_score(y_test, y_pred_before)
    f1_before = f1_score(y_test, y_pred_before)
    roc_auc_before = roc_auc_score(y_test, y_pred_before)
    
    print(f"\n📊 Метрики ДО оптимізації:")
    print(f"   Accuracy: {acc_before:.4f} | F1: {f1_before:.4f} | ROC-AUC: {roc_auc_before:.4f}")
    
    # Оптимізація
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[model_name],
        n_iter=50,
        scoring=target_metric,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\n⏳ Почато оптимізацію...")
    start_time = datetime.now()
    
    if model_name == "Logistic Regression":
        random_search.fit(X_train_scaled, y_train)
        y_pred_after = random_search.best_estimator_.predict(X_test_scaled)
    else:
        random_search.fit(X_train, y_train)
        y_pred_after = random_search.best_estimator_.predict(X_test)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"✅ Завершено за {duration:.1f} секунд")
    
    # Найкращі параметри
    print(f"\n🏆 Найкращі параметри:")
    for param, value in random_search.best_params_.items():
        print(f"   • {param}: {value}")
    
    # Метрики ПІСЛЯ
    acc_after = accuracy_score(y_test, y_pred_after)
    f1_after = f1_score(y_test, y_pred_after)
    roc_auc_after = roc_auc_score(y_test, y_pred_after)
    precision_after = precision_score(y_test, y_pred_after)
    recall_after = recall_score(y_test, y_pred_after)
    
    print(f"\n📊 Метрики ПІСЛЯ оптимізації:")
    print(f"   Accuracy: {acc_after:.4f} | F1: {f1_after:.4f} | ROC-AUC: {roc_auc_after:.4f}")
    print(f"\n📈 Покращення:")
    print(f"   Accuracy: {(acc_after - acc_before)*100:+.2f}%")
    print(f"   F1-Score: {(f1_after - f1_before)*100:+.2f}%")
    print(f"   ROC-AUC:  {(roc_auc_after - roc_auc_before)*100:+.2f}%")
    
    # Збереження результатів
    optimization_results.extend([
        {
            'model_name': model_name,
            'optimization_stage': 'before',
            'accuracy': float(acc_before),
            'f1_score': float(f1_before),
            'roc_auc': float(roc_auc_before),
            'precision': None,
            'recall': None,
            'best_params': None,
            'cv_score': None
        },
        {
            'model_name': model_name,
            'optimization_stage': 'after',
            'accuracy': float(acc_after),
            'f1_score': float(f1_after),
            'roc_auc': float(roc_auc_after),
            'precision': float(precision_after),
            'recall': float(recall_after),
            'best_params': str(random_search.best_params_),
            'cv_score': float(random_search.best_score_)
        }
    ])
    
    # Оновлення моделі
    models[model_name] = random_search.best_estimator_

# Збереження в БД
pd.DataFrame(optimization_results).to_sql(
    'hyperparameter_optimization', engine, if_exists='append', index=False
)
print("\n✅ Результати оптимізації збережено в БД")

# ============================================================================
# 10. ЗБЕРЕЖЕННЯ ОПТИМІЗОВАНИХ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*70)
print("💾 ЗБЕРЕЖЕННЯ ОПТИМІЗОВАНИХ МОДЕЛЕЙ")
print("="*70)

saved_models_info = []

for model_name, model in models.items():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.lower().replace(" ", "_")
    filename = f"models/{safe_name}_optimized_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        if model_name == "Logistic Regression":
            model_package = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names
            }
        else:
            model_package = {
                'model': model,
                'feature_names': feature_names
            }
        pickle.dump(model_package, f)
    
    file_size = os.path.getsize(filename) / 1024
    print(f"✅ {model_name}: {filename} ({file_size:.2f} KB)")
    
    saved_models_info.append({
        'model_name': model_name,
        'filename': filename,
        'file_size_kb': file_size,
        'saved_at': datetime.now()
    })

pd.DataFrame(saved_models_info).to_sql('saved_models', engine, if_exists='append', index=False)

# ============================================================================
# 11. ФІНАЛЬНА ОЦІНКА МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*70)
print("📊 ФІНАЛЬНА ОЦІНКА МОДЕЛЕЙ")
print("="*70)

final_results = []

for model_name, model in models.items():
    print(f"\n🔍 {model_name}:")
    
    if model_name == "Logistic Regression":
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        train_proba = model.predict_proba(X_train_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"   Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"   Test F1: {test_f1:.4f} | Test ROC-AUC: {test_auc:.4f}")
    
    overfit = train_acc - test_acc
    print(f"   Overfit Gap: {overfit:.4f}", end="")
    if overfit < 0.03:
        print(" ✅")
    elif overfit < 0.05:
        print(" ✓")
    else:
        print(" ⚠️")
    
    final_results.append({
        'Model': model_name,
        'Train_Acc': train_acc,
        'Test_Acc': test_acc,
        'Test_F1': test_f1,
        'Test_ROC_AUC': test_auc,
        'Overfit_Gap': overfit
    })

final_df = pd.DataFrame(final_results)
print("\n" + "="*70)
print("🏆 ПІДСУМКОВА ТАБЛИЦЯ")
print("="*70)
print(final_df.to_string(index=False))

best_model_idx = final_df['Test_F1'].idxmax()
best_final_model = final_df.iloc[best_model_idx]

print("\n" + "="*70)
print(f"🥇 НАЙКРАЩА МОДЕЛЬ: {best_final_model['Model']}")
print("="*70)
print(f"   Test F1: {best_final_model['Test_F1']:.4f}")
print(f"   Test ROC-AUC: {best_final_model['Test_ROC_AUC']:.4f}")
print("="*70)

print("\n✅ ВСІ ЕТАПИ ЗАВЕРШЕНО УСПІШНО!")
print(f"📁 Моделі: models/")
print(f"💾 Метрики: PostgreSQL БД '{database_name}'")
print("="*70)