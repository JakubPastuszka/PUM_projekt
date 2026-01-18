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

# Ustawienia wyświetlania
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Ścieżki
RAW_DATA_DIR = "data/raw/kaggle_airline_satisfaction"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_FILE = os.path.join(RAW_DATA_DIR, "test.csv")


def save_processed(df, filename):
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[ZAPIS] Zapisano etap: {path}")


def main():
    if not os.path.exists(TRAIN_FILE):
        print(f"[BŁĄD] Nie znaleziono pliku: {TRAIN_FILE}")
        return

    # --- KROK 0: Wczytanie ---
    print("\n[KROK 0] Wczytywanie danych...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # --- KROK 1: EDA ---
    analyze_dataset(train_df, "Zbiór Treningowy")

    # --- KROK 2: Czyszczenie ---
    print("\n[KROK 1] Czyszczenie (id, Unnamed)...")
    train_df_cleaned = clean_data(train_df)
    test_df_cleaned = clean_data(test_df)

    save_processed(train_df_cleaned, "train_1_cleaned.csv")
    save_processed(test_df_cleaned, "test_1_cleaned.csv")

    # --- KROK 3: Imputacja MICE ---
    print("\n[KROK 2] Imputacja MICE...")
    train_df_imputed = impute_missing_values(train_df_cleaned)
    test_df_imputed = impute_missing_values(test_df_cleaned)

    inspect_imputation_results(train_df_cleaned, train_df_imputed)

    save_processed(train_df_imputed, "train_2_imputed.csv")
    save_processed(test_df_imputed, "test_2_imputed.csv")

    # --- INSPEKCJA LOGICZNA ---
    inspect_logical_consistency(train_df_imputed)

    # --- KROK 4: Feature Engineering ---
    print("\n[KROK 3] Feature Engineering (Naprawa zer, Kodowanie, Normalizacja, Average Score)...")

    train_df_featured = process_features(train_df_imputed)
    test_df_featured = process_features(test_df_imputed)

    # --- WERYFIKACJA KOŃCOWA ---
    print("\n" + "=" * 50)
    print("[WERYFIKACJA KOŃCOWA] Czy normalizacja i nowe cechy działają?")
    print("=" * 50)

    # 1. Sprawdźmy czy dane są znormalizowane (Max powinno być 1.0)
    print("\n[INFO] Sprawdzanie zakresów wartości (powinno być 0-1 dla znormalizowanych):")
    # Sprawdzamy jeśli te kolumny istnieją (zabezpieczenie)
    cols_to_check = ['Age', 'Total Delay', 'Average Service Score']
    existing_check_cols = [c for c in cols_to_check if c in train_df_featured.columns]
    print(train_df_featured[existing_check_cols].describe().loc[['min', 'max']])

    # 2. Sprawdźmy korelację naszej nowej zmiennej 'Average Service Score'
    if 'Average Service Score' in train_df_featured.columns:
        print("\n[INFO] Korelacja 'Average Service Score' z satysfakcją:")
        corr = train_df_featured['Average Service Score'].corr(train_df_featured['satisfaction'])
        print(f"Korelacja wynosi: {corr:.4f}")

    # 3. Weryfikacja techniczna
    print("\n[WERYFIKACJA] Czy zniknęły obiekty tekstowe?")
    obj_cols = train_df_featured.select_dtypes(include=['object']).columns.tolist()
    print(f"Kolumny tekstowe (powinno być pusto): {obj_cols}")

    # Zapis finalny
    save_processed(train_df_featured, "train_3_featured.csv")
    save_processed(test_df_featured, "test_3_featured.csv")

    print("\n[SUKCES] Dane gotowe do modelowania!")


if __name__ == "__main__":
    main()