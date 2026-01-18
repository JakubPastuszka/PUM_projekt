import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def process_features(df):
    """
    Realizuje Feature Engineering:
    1. Naprawia 'zera' w ocenach (0 -> Mediana).
    2. Usuwa zbędne kolumny.
    3. Tworzy nowe cechy (Total Delay, Average Score).
    4. Normalizuje dane numeryczne (MinMax).
    5. Koduje zmienne tekstowe (One-Hot i Ordinal).
    """
    df_out = df.copy()

    print("[FEATURES] Rozpoczynanie transformacji danych...")

    # --- 1. Obsługa logiczna zer (0 -> NaN -> Mediana) ---
    cols_with_zeros = [
        'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Online boarding', 'Leg room service',
        'Food and drink', 'Cleanliness'
    ]
    existing_zeros_cols = [c for c in cols_with_zeros if c in df_out.columns]

    for col in existing_zeros_cols:
        # Obliczamy medianę tylko z wartości niezerowych
        median_val = df_out[df_out[col] != 0][col].median()
        # Zamieniamy 0 na tę medianę
        df_out[col] = df_out[col].replace(0, median_val)

    # --- 2. Feature Selection ---
    cols_to_drop = ['Gate location', 'Departure/Arrival time convenient']
    cols_to_drop = [c for c in cols_to_drop if c in df_out.columns]

    if cols_to_drop:
        df_out.drop(columns=cols_to_drop, inplace=True)
        print(f"Usunięto mało istotne kolumny: {cols_to_drop}")

    # --- 3. Feature Engineering ---

    # A. Total Delay
    if 'Departure Delay in Minutes' in df_out.columns and 'Arrival Delay in Minutes' in df_out.columns:
        df_out['Total Delay'] = df_out['Departure Delay in Minutes'] + df_out['Arrival Delay in Minutes']
        df_out.drop(columns=['Arrival Delay in Minutes', 'Departure Delay in Minutes'], inplace=True)

    # B. Average Service Score (Średnia ocena)
    rating_cols = [
        'Inflight wifi service', 'Ease of Online booking', 'Online boarding',
        'Seat comfort', 'Inflight entertainment', 'On-board service',
        'Leg room service', 'Baggage handling', 'Checkin service',
        'Inflight service', 'Cleanliness', 'Food and drink'
    ]
    existing_rating_cols = [c for c in rating_cols if c in df_out.columns]

    if existing_rating_cols:
        df_out['Average Service Score'] = df_out[existing_rating_cols].mean(axis=1)

    # C. Kategoryzacja Dystansu
    if 'Flight Distance' in df_out.columns:
        df_out['Flight Distance Category'] = pd.cut(
            df_out['Flight Distance'],
            bins=[-1, 800, 2500, 100000],
            labels=[0, 1, 2]
        ).astype(int)

    # --- 4. Normalizacja MinMax ---
    # Normalizujemy zmienne ciągłe do zakresu 0-1
    cols_to_normalize = ['Age', 'Flight Distance', 'Total Delay', 'Average Service Score']
    cols_to_normalize = [c for c in cols_to_normalize if c in df_out.columns]

    if cols_to_normalize:
        scaler = MinMaxScaler()
        df_out[cols_to_normalize] = scaler.fit_transform(df_out[cols_to_normalize])
        print(f"[INFO] Znormalizowano kolumny (MinMax): {cols_to_normalize}")

    # --- 5. Kodowanie Zmiennych ---

    # Target (Satisfaction)
    if 'satisfaction' in df_out.columns:
        satisfaction_map = {'neutral or dissatisfied': 0, 'satisfied': 1}
        if df_out['satisfaction'].dtype == 'object':
            df_out['satisfaction'] = df_out['satisfaction'].map(satisfaction_map)

    # Class (Ordinal)
    class_map = {'Eco': 1, 'Eco Plus': 2, 'Business': 3}
    if 'Class' in df_out.columns and df_out['Class'].dtype == 'object':
        df_out['Class'] = df_out['Class'].map(class_map)

    # One-Hot Encoding
    cols_to_encode = ['Gender', 'Customer Type', 'Type of Travel']
    cols_to_encode = [c for c in cols_to_encode if c in df_out.columns]

    df_out = pd.get_dummies(df_out, columns=cols_to_encode, drop_first=True)

    # Konwersja bool -> int
    for col in df_out.columns:
        if df_out[col].dtype == 'bool':
            df_out[col] = df_out[col].astype(int)

    return df_out