import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- FUNKCJA WSPÓLNA (POMOCNICZA) ---
def load_and_split_data(df_path):
    """
    Tej funkcji używa każdy z Was. Wczytuje dane i dzieli je na X i y.
    """
    print(f"[DATA] Wczytywanie danych z {df_path}...")
    df = pd.read_csv(df_path)

    target_col = 'satisfaction'
    if target_col not in df.columns:
        raise ValueError("Brak kolumny 'satisfaction'!")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Podział 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_val, y_train, y_val


# ==========================================
# OSOBA 1: REGRESJA LOGISTYCZNA
# ==========================================
def run_logistic_regression(df_path, models_dir="models", output_dir="outputs"):
    print("\n" + "=" * 40)
    print(" [OSOBA 1] Trenowanie: Regresja Logistyczna")
    print("=" * 40)

    # 1. Pobranie danych
    X_train, X_val, y_train, y_val = load_and_split_data(df_path)

    # 2. Model
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Ewaluacja
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    # 4. Zapis
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, "model_logreg.joblib"))

    # 5. Wykres unikalny dla Regresji: Wpływ cech (Współczynniki)
    # Pokazuje, które cechy ciągną wynik w górę (pozytywne), a które w dół (negatywne)
    coefs = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': clf.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coefs.head(10))  # Top 10 pozytywnych
    plt.title('Regresja Logistyczna: Co najbardziej zwiększa zadowolenie?')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_logreg_coefs.png"))
    print("[INFO] Zapisano wykres współczynników.")


# ==========================================
# OSOBA 2: DRZEWO DECYZYJNE
# ==========================================
def run_decision_tree(df_path, models_dir="models", output_dir="outputs"):
    print("\n" + "=" * 40)
    print(" [OSOBA 2] Trenowanie: Drzewo Decyzyjne")
    print("=" * 40)

    # 1. Pobranie danych
    X_train, X_val, y_train, y_val = load_and_split_data(df_path)

    # 2. Model (max_depth=4 żeby wykres był czytelny)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Ewaluacja
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    # 4. Zapis
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, "model_tree.joblib"))

    # 5. Wykres unikalny dla Drzewa: Wizualizacja Decyzji
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X_train.columns, class_names=['Neutral/Dissatisfied', 'Satisfied'], filled=True,
              fontsize=10)
    plt.title('Drzewo Decyzyjne (pierwsze 4 poziomy)')
    plt.savefig(os.path.join(output_dir, "plot_tree_viz.png"))
    print("[INFO] Zapisano wizualizację drzewa.")


# ==========================================
# OSOBA 3: LAS LOSOWY (RANDOM FOREST)
# ==========================================
def run_random_forest(df_path, models_dir="models", output_dir="outputs"):
    print("\n" + "=" * 40)
    print(" [OSOBA 3] Trenowanie: Random Forest")
    print("=" * 40)

    # 1. Pobranie danych
    X_train, X_val, y_train, y_val = load_and_split_data(df_path)

    # 2. Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 3. Ewaluacja
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    # 4. Zapis
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, "model_forest.joblib"))

    # 5. Wykres unikalny dla Lasu: Ważność Cech (Feature Importance)
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df.head(10))
    plt.title('Random Forest: Ranking najważniejszych cech')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_forest_importance.png"))
    print("[INFO] Zapisano ranking ważności cech.")

