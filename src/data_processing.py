import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def analyze_dataset(df, name="Dataset"):
    # Ustawienia wyświetlania
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(f"\n--- ANALIZA EDA: {name} ---")

    # Podstawowe metryki
    print("\n[INFO] Podstawowe metryki (min, max, mean, median):")
    stats = df.describe().loc[['min', 'max', 'mean', '50%']]
    stats.rename(index={'50%': 'median'}, inplace=True)
    print(stats)

    # Szukanie braków danych
    print("\n[INFO] Brakujace wartosci (NaN):")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Brak brakujacych wartosci.")

    # Typy danych
    print("\n[INFO] Typy danych:")
    print(df.dtypes)

    return stats


def clean_data(df):
    """Usuwa kolumny, ktore nie sa predyktorami (id, Unnamed)."""
    cols_to_drop = ['Unnamed: 0', 'id']
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_cols)


def impute_missing_values(df):
    """
    Wykonywanie imputacji MICE dla kolumn numerycznych.
    Zwraca DataFrame z uzupelnionymi brakami.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # Inicjalizacja IterativeImputer (MICE)
    mice_imputer = IterativeImputer(random_state=42)

    # Dopasowanie i transformacja
    imputed_data = mice_imputer.fit_transform(df_numeric)

    # Stworzenie DataFrame z wynikow
    df_imputed_numeric = pd.DataFrame(
        imputed_data,
        columns=numeric_cols,
        index=df.index
    )

    # Podmiana kolumn w oryginalnym DataFrame
    df_final = df.copy()
    for col in numeric_cols:
        df_final[col] = df_imputed_numeric[col]

    return df_final


def inspect_imputation_results(df_original, df_imputed):
    """
    Porównuje dane przed i po imputacji MICE dla kolumny Arrival Delay.
    Wyświetla w terminalu przykłady uzupełnionych wartości.
    """
    print("\n[INSPEKCJA] Sprawdzanie efektywności MICE:")

    if 'Arrival Delay in Minutes' not in df_original.columns:
        print("Brak kolumny 'Arrival Delay in Minutes' w danych źródłowych.")
        return

    # Znajdujemy indeksy wierszy, gdzie w oryginale było NaN
    missing_indices = df_original[df_original['Arrival Delay in Minutes'].isnull()].index

    if not missing_indices.empty:
        comparison = pd.DataFrame({
            'Departure Delay': df_original.loc[missing_indices, 'Departure Delay in Minutes'],
            'Original Arrival': df_original.loc[missing_indices, 'Arrival Delay in Minutes'],
            'Imputed Arrival': df_imputed.loc[missing_indices, 'Arrival Delay in Minutes']
        })
        print(comparison.head(5))
        print(f"Średnia wstawionych wartości: {comparison['Imputed Arrival'].mean():.2f}")
    else:
        print("Brak wartości NaN w źródle do porównania.")


def inspect_logical_consistency(df):
    """
    Sprawdza logikę danych zgodnie z dyskusją z Kaggle:
    1. Zlicza wystąpienia zer w kolumnach ocen (podejrzenie 'Not Applicable').
    2. Sprawdza korelację zmiennych z satysfakcją.
    """
    print("\n" + "=" * 50)
    print("[INSPEKCJA LOGICZNA] Analiza spójności i korelacji")
    print("=" * 50)

    # A. Sprawdzanie zer
    print("\n[ANALIZA ZER] Liczba ocen '0' (Podejrzewane jako 'Nie dotyczy'):")
    rating_cols = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
    ]
    # Filtrujemy tylko te kolumny, które istnieją w DF
    available_cols = [c for c in rating_cols if c in df.columns]
    zeros_check = (df[available_cols] == 0).sum()
    print(zeros_check[zeros_check > 0])

    # B. Sprawdzanie korelacji
    print("\n[ANALIZA KORELACJI] Wpływ zmiennych na Satisfaction:")
    if 'satisfaction' in df.columns:
        temp_corr_df = df.copy()
        # Tymczasowe kodowanie do sprawdzenia korelacji
        if temp_corr_df['satisfaction'].dtype == 'object':
            temp_corr_df['target_numeric'] = temp_corr_df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
        else:
            temp_corr_df['target_numeric'] = temp_corr_df['satisfaction']

        correlations = temp_corr_df.select_dtypes(include=[np.number]).corr()['target_numeric'].sort_values(
            ascending=False)
        print(correlations.drop('target_numeric', errors='ignore'))

        print("\n[INFO] Ujemna korelacja dla opóźnień jest poprawna (większe opóźnienie = mniejsza satysfakcja).")
    else:
        print("Brak kolumny 'satisfaction' - pomijam analizę korelacji.")