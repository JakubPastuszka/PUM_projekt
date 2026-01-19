import pandas as pd
import os
import numpy as np

# Importujemy funkcje z Twoich modułów src
from src.data_processing import (
    analyze_dataset,
    clean_data,
    impute_missing_values,
    inspect_imputation_results,
    inspect_logical_consistency
)
from src.features import process_features

# Importujemy 3 funkcje modelowania
from src.model_building import (
    run_logistic_regression,
    run_decision_tree,
    run_random_forest,
)

# --- KONFIGURACJA I STAŁE ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

RAW_DATA_DIR = "data/raw/kaggle_airline_satisfaction"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_FILE = os.path.join(RAW_DATA_DIR, "test.csv")
FINAL_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_3_featured.csv")


# --- FUNKCJE POMOCNICZE I ETAPY ---

def save_processed(df, filename):
    """Zapisuje DataFrame do CSV i informuje o tym w konsoli."""
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[ZAPIS] Zapisano etap: {path}")


def load_raw_data():
    """Krok 0: Wczytanie surowych danych."""
    if not os.path.exists(TRAIN_FILE):
        print(f"[BŁĄD] Nie znaleziono pliku: {TRAIN_FILE}")
        return None, None

    print("\n[KROK 0] Wczytywanie danych...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def run_eda_step(df):
    """Krok 1: Analiza eksploracyjna danych (EDA)."""
    analyze_dataset(df, "Zbiór Treningowy")


def run_cleaning_step(train_df, test_df):
    """Krok 2: Czyszczenie podstawowe (usuwanie id, Unnamed)."""
    print("\n[KROK 1] Czyszczenie (id, Unnamed)...")
    train_cleaned = clean_data(train_df)
    test_cleaned = clean_data(test_df)

    save_processed(train_cleaned, "train_1_cleaned.csv")
    save_processed(test_cleaned, "test_1_cleaned.csv")
    return train_cleaned, test_cleaned


def run_imputation_step(train_df, test_df):
    """Krok 3: Imputacja brakujących wartości (MICE)."""
    print("\n[KROK 2] Imputacja MICE...")
    train_imputed = impute_missing_values(train_df)
    test_imputed = impute_missing_values(test_df)

    # Inspekcja wyników
    inspect_imputation_results(train_df, train_imputed)
    inspect_logical_consistency(train_imputed)

    save_processed(train_imputed, "train_2_imputed.csv")
    save_processed(test_imputed, "test_2_imputed.csv")
    return train_imputed, test_imputed


def run_feature_engineering_step(train_df, test_df):
    """Krok 4: Inżynieria cech, normalizacja i weryfikacja."""
    print("\n[KROK 3] Feature Engineering (Naprawa zer, Kodowanie, Normalizacja, Average Score)...")

    train_featured = process_features(train_df)
    test_featured = process_features(test_df)

    # --- WERYFIKACJA WEWNĘTRZNA ---
    print("\n" + "=" * 50)
    print("[WERYFIKACJA KOŃCOWA] Czy normalizacja i nowe cechy działają?")
    print("=" * 50)

    # 1. Zakresy wartości
    cols_to_check = ['Age', 'Total Delay', 'Average Service Score']
    existing_check_cols = [c for c in cols_to_check if c in train_featured.columns]
    print("\n[INFO] Zakresy wartości (oczekiwane 0-1):")
    print(train_featured[existing_check_cols].describe().loc[['min', 'max']])

    # 2. Korelacja Average Score
    if 'Average Service Score' in train_featured.columns:
        corr = train_featured['Average Service Score'].corr(train_featured['satisfaction'])
        print(f"\n[INFO] Korelacja 'Average Service Score' z satysfakcją: {corr:.4f}")

    # 3. Sprawdzenie tekstów
    obj_cols = train_featured.select_dtypes(include=['object']).columns.tolist()
    print(f"\n[WERYFIKACJA] Kolumny tekstowe (powinno być pusto): {obj_cols}")

    save_processed(train_featured, "train_3_featured.csv")
    save_processed(test_featured, "test_3_featured.csv")

    return train_featured, test_featured


def run_modeling_pipeline(train_data_path):
    """Krok 5: Uruchomienie treningu modeli (podział na osoby)."""
    print("\n" + "#" * 60)
    print(" ROZPOCZYNAMY TRENOWANIE MODELI (3 OSOBY)")
    print("#" * 60)

    # 1. Osoba od Regresji
    run_logistic_regression(train_data_path)

    # 2. Osoba od Drzew
    run_decision_tree(train_data_path)

    # 3. Osoba od Random Forest
    run_random_forest(train_data_path)

    print("\n[SUKCES] Wszystkie modele wytrenowane i zapisane w folderze 'models/'!")


# --- GŁÓWNA FUNKCJA ---

def main():
    # 1. Wczytanie
    train_df, test_df = load_raw_data()
    if train_df is None:
        return

    # 2. Analiza EDA
    run_eda_step(train_df)

    # 3. Czyszczenie
    train_clean, test_clean = run_cleaning_step(train_df, test_df)

    # 4. Imputacja
    train_imp, test_imp = run_imputation_step(train_clean, test_clean)

    # 5. Feature Engineering
    # Zwracamy zmienne, ale też zapisujemy pliki na dysk, które modelowanie odczyta
    train_feat, test_feat = run_feature_engineering_step(train_imp, test_imp)

    print("\n[SUKCES] Dane przetworzone i gotowe do modelowania!")

    # 6. Modelowanie
    run_modeling_pipeline(FINAL_TRAIN_PATH)


if __name__ == "__main__":
    main()