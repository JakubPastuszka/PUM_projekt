import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from shap.benchmark.methods import coef
from sklearn import calibration
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve, \
    auc, precision_recall_fscore_support


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
def run_logistic_regression(
    df_path,
    models_dir="models/log_reg",
    output_dir="outputs/log_reg",
    pairwise_poly=False,
    pairwise_poly_top10=False,
    pairwise_linear=True,
    random_search=True,
    calibration=True
):

    import os
    import joblib
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix, roc_curve
    )
    from sklearn.model_selection import ParameterSampler

    # ==================================================
    # Helper: threshold optimization
    # ==================================================
    def find_best_threshold(y_true, y_prob):
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t, best_acc = 0.5, -1
        for t in thresholds:
            acc = accuracy_score(y_true, (y_prob >= t).astype(int))
            if acc > best_acc:
                best_acc, best_t = acc, t
        return best_t, best_acc, thresholds

    # ==================================================
    # Prepare directories
    # ==================================================
    search_type = "random" if random_search else "grid"
    suffix = "poly/top" if pairwise_poly and pairwise_poly_top10 else "poly" if pairwise_poly else "linear"
    models_dir = f"{models_dir}/{suffix}/{search_type}"
    output_dir = f"{output_dir}/{suffix}/{search_type}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 40)
    print(" [RETRAIN] Logistic Regression (FULL DATA)")
    print("=" * 40)

    # ==================================================
    # 1. Load data
    # ==================================================
    df = pd.read_csv(df_path)
    if "satisfaction" not in df.columns:
        raise ValueError("Brak kolumny 'satisfaction'")

    X = df.drop(columns=["satisfaction"])
    y = df["satisfaction"]

    # ==================================================
    # 2. Scaling
    # ==================================================
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"[INFO] Scaling zakończony – {X.shape[1]} cech")

    # ==================================================
    # 3. Pairwise linear
    # ==================================================
    if pairwise_linear:
        from itertools import combinations
        max_pairs = 105
        pairs = list(combinations(X.columns, 2))[:max_pairs]
        for f1, f2 in pairs:
            X[f"{f1}_plus_{f2}"] = X[f1] + X[f2]
            X[f"{f1}_minus_{f2}"] = X[f1] - X[f2]
        print(f"[INFO] Dodano pairwise linear features → {X.shape[1]} cech")

    # ==================================================
    # 4. Pairwise polynomial
    # ==================================================
    if pairwise_poly:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X = pd.DataFrame(
            poly.fit_transform(X),
            columns=poly.get_feature_names_out(X.columns)
        )
        print(f"[INFO] Dodano pairwise polynomial features → {X.shape[1]} cech")

    # ==================================================
    # 5. Top 10 features (opcjonalne)
    # ==================================================
    if pairwise_poly and pairwise_poly_top10:
        temp = LogisticRegression(solver="saga", max_iter=10000, random_state=42)
        temp.fit(X, y)
        coef_tmp = pd.DataFrame({
            "Feature": X.columns,
            "Coef": temp.coef_[0]
        })
        top_feats = (
            coef_tmp.assign(abs_coef=np.abs(coef_tmp.Coef))
            .sort_values("abs_coef", ascending=False)
            .head(10)["Feature"]
        )
        X = X[top_feats.tolist()]
        print(f"[INFO] Zredukowano do TOP 10 cech")

    # ==================================================
    # 6. Parameter sampling
    # ==================================================
    param_rand = {
        "C": [0.01, 0.1, 1, 10, 50],
        "l1_ratio": [0, 0.5, 1],
        "class_weight": [None, "balanced"]
    }

    params_list = list(ParameterSampler(param_rand, n_iter=3, random_state=42))
    results = []

    # ==================================================
    # 7. ITERACJE (nie foldy)
    # ==================================================
    for i, p in enumerate(params_list, 1):

        print("\n" + "=" * 40)
        print(f"[ITERACJA {i}] Parametry: {p}")
        print("=" * 40)

        base_model = LogisticRegression(
            solver="saga",
            max_iter=10000,
            random_state=42,
            **p
        )
        base_model.fit(X, y)

        # ===== Calibration =====
        if calibration:
            model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
            model.fit(X, y)
        else:
            model = base_model

        # ===== Predictions =====
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        acc_iter = accuracy_score(y, y_pred)
        f1_iter = f1_score(y, y_pred)
        auc_iter = roc_auc_score(y, y_prob)

        print(f"Accuracy: {acc_iter:.4f}, F1: {f1_iter:.4f}, AUC: {auc_iter:.4f}")

        best_t, best_acc, thresholds = find_best_threshold(y, y_prob)

        # ==================================================
        # 8. WYKRESY CECH — POPRAWIONE
        # ==================================================
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coef": base_model.coef_[0]
        })
        coef_df["abs_coef"] = coef_df["Coef"].abs()

        # --- Top Positive ---
        top_pos = coef_df.sort_values("Coef", ascending=False).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Coef", y="Feature", data=top_pos)
        plt.title("Top Positive Coefficients")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_pos_iter{i}.png")
        plt.close()

        # --- Top Negative ---
        top_neg = coef_df.sort_values("Coef", ascending=True).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Coef", y="Feature", data=top_neg)
        plt.title("Top Negative Coefficients")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_neg_iter{i}.png")
        plt.close()

        # --- Global importance ---
        importance = coef_df.sort_values("abs_coef", ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        sns.barplot(x="abs_coef", y="Feature", data=importance)
        plt.title("Coefficient Importance (|coef|)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/coef_importance_iter{i}.png")
        plt.close()

        # ==================================================
        # 9. Confusion matrix
        # ==================================================
        y_pred_best = (y_prob >= best_t).astype(int)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(y, y_pred_best),
            annot=True, fmt="d", cmap="Blues"
        )
        plt.title(f"Confusion Matrix – iteracja {i}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cm_iter{i}.png")
        plt.close()

        # ==================================================
        # 10. ROC
        # ==================================================
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc_iter:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_iter{i}.png")
        plt.close()

        # ==================================================
        # 11. Save model + results
        # ==================================================
        joblib.dump(model, f"{models_dir}/model_iter{i}.joblib")

        results.append({
            **p,
            "acc_0.5": accuracy_score(y, (y_prob >= 0.5).astype(int)),
            "acc_best_t": best_acc,
            "best_t": best_t,
            "AUC": auc_iter
        })

    # ==================================================
    # 12. Save tuning results
    # ==================================================
    pd.DataFrame(results).to_csv(
        f"{models_dir}/tuning_results.csv",
        index=False
    )
    print(f"\n[INFO] Wyniki zapisane → {models_dir}/tuning_results.csv")



def run_logistic_regression_kfold(
    df_path,
    models_dir=None,
    output_dir=None,
    k_fold = False,
    n_splits=5,
    pairwise_poly = True, # Trzeba zmienic dla Kfold ale trwa to z godzine, moze 2
    pairwise_poly_top10 = False, # Ograniczenie polynomial do 10 cech
    pairwise_linear = False, #
    random_search = True
):
    """
    Regresja logistyczna trenowana z użyciem K-Fold (Stratified) z iteracją po kombinacjach parametrów.
    Dane wejściowe: CSV po pełnym preprocessing'u.
    """

    print("\n" + "=" * 40)
    print(" [OSOBA 1] Regresja Logistyczna (5-Fold CV + Tuning)")
    print("=" * 40)

    search_type = "random" if random_search else "grid"

    if pairwise_poly:
        if pairwise_poly_top10:
            models_dir = f"models/log_reg/kfold/poly/top/{search_type}"
            output_dir = f"outputs/log_reg/kfold/poly/top/{search_type}"
        else:
            models_dir = f"models/log_reg/kfold/poly/{search_type}"
            output_dir = f"outputs/log_reg/kfold/poly/{search_type}"
    else:
        models_dir = f"models/log_reg/kfold/linear/{search_type}"
        output_dir = f"outputs/log_reg/kfold/linear/{search_type}"

    # --- 1. Wczytanie danych ---
    df = pd.read_csv(df_path)

    if 'satisfaction' not in df.columns:
        raise ValueError("Brak kolumny 'satisfaction'!")

    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # =====  PAIRWISE LINEAR  =====
    if pairwise_linear:
        max_pairs = 105  # ograniczenie liczby par, żeby nie wydłużać czasu trenowania
        from itertools import combinations
        feature_pairs = list(combinations(X.columns, 2))[:max_pairs]

        for f1, f2 in feature_pairs:
            X[f"{f1}_plus_{f2}"] = X[f1] + X[f2]
            X[f"{f1}_minus_{f2}"] = X[f1] - X[f2]
        print(f"[INFO] Dodano pairwise additive features: +{2 * len(feature_pairs)} cech, nowa liczba cech = {X.shape[1]}")


    # ===== PAIRWISE POLYNOMIAL =====
    if pairwise_poly:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        X = pd.DataFrame(X_poly, columns=feature_names)
        print(f"[INFO] Dodano pairwise features: nowa liczba cech = {X.shape[1]}")

    if pairwise_poly_top10:
        temp_model = LogisticRegression(solver="saga", max_iter=10000, random_state=42)
        temp_model.fit(X, y)
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": temp_model.coef_[0]
        })
        coef_df["abs_coef"] = coef_df["Coefficient"].abs()
        top_features = coef_df.sort_values("abs_coef", ascending=False).head(10)["Feature"].tolist()
        X = X[top_features]  # zostawiamy tylko 10 najlepszych cech
        print(f"[INFO] Zredukowano do 10 najlepszych cech po pairwise polynomial: {top_features}")

    # --- 2. Definicja siatki parametrów ---
    param_grid = [
        {"C": 100.0, "l1_ratio": 1.0, "class_weight": None},
        {"C": 10.0, "l1_ratio": 1.0, "class_weight": "balanced"},
        # {"C": 1.0, "l1_ratio": 0.0, "class_weight": None},
        # {"C": 0.1, "l1_ratio": 0.0, "class_weight": None},
        # {"C": 10.0, "l1_ratio": 0.0, "class_weight": None},
        # {"C": 1.0, "l1_ratio": 0.5, "class_weight": None},
        # {"C": 1.0, "l1_ratio": 1.0, "class_weight": None},
        # {"C": 1.0, "l1_ratio": 0.0, "class_weight": "balanced"},
        # {"C": 0.5, "l1_ratio": 0.5, "class_weight": "balanced"},
    ]


    # ==== Random Search ====
    param_rand = {
        "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "class_weight": [None, "balanced"]
    }

    n_iter_search = 3  # liczba losowych kombinacji
    np.random.seed(42)  # seed dla powtarzalności
    param_samples = list(ParameterSampler(param_rand, n_iter=n_iter_search, random_state=42))
    # =======================

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # --- 3. Iteracja po parametrach ---
    param_list = param_samples if random_search else param_grid

    for i, params in enumerate(param_list, 1):
        print("\n" + "="*40)
        print(f"[TEST {i}] Parametry: {params}")
        print("="*40)

        model = LogisticRegression(
            C=params["C"],
            solver="saga",
            l1_ratio=params["l1_ratio"],
            class_weight=params["class_weight"],
            max_iter=10000,
            random_state=42
        )

        # --- K-Fold CV ---
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc_list, f1_list, auc_list = [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n[FOLD {fold}]")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            acc_list.append(accuracy_score(y_val, y_pred))
            f1_list.append(f1_score(y_val, y_pred))
            auc_list.append(roc_auc_score(y_val, y_prob))

            print(f"Fold Accuracy: {acc_list[-1]:.4f}, F1: {f1_list[-1]:.4f}, AUC: {auc_list[-1]:.4f}")

        mean_acc = np.mean(acc_list)
        mean_f1 = np.mean(f1_list)
        mean_auc = np.mean(auc_list)
        print(f"\n[ŚREDNIA K-FOLD] Accuracy: {mean_acc:.4f}, F1: {mean_f1:.4f}, AUC: {mean_auc:.4f}")

        results.append({
            "C": params["C"],
            "l1_ratio": params["l1_ratio"],
            "class_weight": params["class_weight"],
            "Accuracy": mean_acc,
            "F1": mean_f1,
            "AUC": mean_auc
        })

        # --- Trenowanie finalnego modelu ---
        model.fit(X, y)
        model_filename = f"model_C{params['C']}_l1{params['l1_ratio']}_cw{params['class_weight']}.joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        print(f"[INFO] Finalny model zapisany do {model_filename}")

        # --- Wyjaśnialność: współczynniki ---
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })

        coef_importance = coef_df.copy()
        coef_importance["abs_coef"] = coef_importance["Coefficient"].abs()
        coef_importance = coef_importance.sort_values("abs_coef", ascending=False)
        coef_signed = coef_df.sort_values("Coefficient", ascending=False)


        # ===== TOP INTERAKCJE =====
        if pairwise_poly or max_pairs > 0:
            interaction_df = coef_df[
                coef_df["Feature"].str.contains(" ") | coef_df["Feature"].str.contains("_plus_") | coef_df["Feature"].str.contains("_minus_")
            ].copy()

            top_interactions = coef_importance.head(10)

            print("\n[TOP 10 INTERAKCJI]")
            print(top_interactions[["Feature", "Coefficient"]])

            plt.figure(figsize=(9, 6))
            sns.barplot(
                x="Coefficient",
                y="Feature",
                data=top_interactions,
                orient="h"
            )
            plt.title(f"Top 10 najważniejszych cech / interakcji – test {i}")
            plt.xlabel("|Coefficient|")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"logreg_top_interactions_test{i}.png"
                )
            )
            plt.close()

        # Top 10 pozytywne
        top_positive = coef_signed.head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Coefficient", y="Feature", data=top_positive, orient="h")
        plt.title(f"Top 10 cech zwiększających satysfakcję – test {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"logreg_top_positive_test{i}.png"))
        plt.close()

        # Top 10 negatywne
        top_negative = coef_signed.tail(10).sort_values(by="Coefficient")
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Coefficient", y="Feature", data=top_negative, orient="h")
        plt.title(f"Top 10 cech obniżających satysfakcję – test {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"logreg_top_negative_test{i}.png"))
        plt.close()

        # --- Confusion Matrix ---
        y_pred_full = model.predict(X)
        cm = confusion_matrix(y, y_pred_full)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neutral/Dissatisfied", "Satisfied"],
                    yticklabels=["Neutral/Dissatisfied", "Satisfied"])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix – test {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"logreg_confusion_matrix_test{i}.png"))
        plt.close()

        # --- ROC + AUC ---
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – test {i}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"logreg_roc_curve_test{i}.png"))
        plt.close()

        # --- Precision / Recall / F1 per klasa ---
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_full, labels=[0, 1])
        metrics_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-score"],
            "Neutral/Dissatisfied": [precision[0], recall[0], f1[0]],
            "Satisfied": [precision[1], recall[1], f1[1]]
        }).set_index("Metric")
        metrics_df.plot(kind="bar", figsize=(8, 5))
        plt.title(f"Precision / Recall / F1 – test {i}")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.legend(title="Klasa")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"logreg_class_metrics_test{i}.png"))
        plt.close()

        # --- Scatter plot: Average Service Score vs P(Satisfaction=1) ---
        feature_name = "Average Service Score"
        threshold_val = 0.5
        if feature_name in X.columns:
            probs = model.predict_proba(X)[:, 1]
            plt.figure(figsize=(7, 5))
            plt.scatter(X[feature_name], probs, c=y, cmap="coolwarm", alpha=0.4)
            plt.axhline(y=threshold_val, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold_val}")
            plt.xlabel(feature_name)
            plt.ylabel("P(Satisfaction = 1)")
            plt.title(f"{feature_name} vs P(satisfaction) – test {i}")
            plt.colorbar(label="True class")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"logreg_scatter_{feature_name.replace(' ', '_')}_test{i}.png"))
            plt.close()

    # --- Zapis wszystkich wyników do CSV ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(models_dir, "tuning_results.csv"), index=False)
    print(f"\n[INFO] Wszystkie modele i wyniki zapisane w '{os.path.join(models_dir, 'tuning_results.csv')}'")


def run_logistic_regression_incremental(
    full_train_path,
    batch_size=25,
    models_dir="models/log_reg_inc",
    output_dir="outputs/log_reg_inc",
    pairwise_poly=True,
    pairwise_poly_top10=False,
    pairwise_linear=False,
    calibration=True
):

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(full_train_path)
    if "satisfaction" not in full_df.columns:
        raise ValueError("Brak kolumny 'satisfaction'")

    used_path = os.path.join(models_dir, "train_used.csv")

    if os.path.exists(used_path):
        used_df = pd.read_csv(used_path)
        new_df = full_df.loc[~full_df.index.isin(used_df.index)]
        if new_df.empty:
            print("[INFO] Brak nowych danych – incremental learning pominięty.")
            return
    else:
        new_df = full_df.copy()

    X = new_df.drop(columns=["satisfaction"])
    y = new_df["satisfaction"].values

    # ==================================================
    # Scaling + feature engineering

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if pairwise_linear:
        from itertools import combinations
        max_pairs = 105
        pairs = list(combinations(X.columns, 2))[:max_pairs]
        for f1, f2 in pairs:
            X[f"{f1}_plus_{f2}"] = X[f1] + X[f2]
            X[f"{f1}_minus_{f2}"] = X[f1] - X[f2]

    if pairwise_poly:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X = pd.DataFrame(
            poly.fit_transform(X),
            columns=poly.get_feature_names_out(X.columns)
        )

    if pairwise_poly and pairwise_poly_top10:
        temp = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
        temp.fit(X, y)
        coef_df = pd.DataFrame({"Feature": X.columns, "Coef": temp.coef_[0]})
        top_feats = (
            coef_df.assign(abs=np.abs(coef_df.Coef))
            .sort_values("abs", ascending=False)
            .head(10)["Feature"]
            .tolist()
        )
        X = X[top_feats]

    # ==================================================
    # Base incremental model
    # ==================================================
    base_model_path = os.path.join(models_dir, "base_model_sgd.joblib")
    calibrated_model_path = os.path.join(models_dir, "calibrated_model.joblib")
    classes = np.unique(full_df["satisfaction"])

    if os.path.exists(base_model_path):
        base_model = joblib.load(base_model_path)
        print("[INFO] Wczytano istniejący model SGD.")
    else:
        base_model = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        print("[INFO] Utworzono nowy model SGD.")

    # ==================================================
    # Incremental batch
    # ==================================================
    results = []
    n_batches = int(np.ceil(len(X) / batch_size))

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X))
        X_batch = X.iloc[start:end]
        y_batch = y[start:end]

        base_model.partial_fit(X_batch, y_batch, classes=classes)

        # --- batch metrics ---
        y_pred = base_model.predict(X_batch)
        y_prob = base_model.predict_proba(X_batch)[:, 1]

        acc = accuracy_score(y_batch, y_pred)
        f1 = f1_score(y_batch, y_pred)
        auc = roc_auc_score(y_batch, y_prob)

        print(f"[BATCH {i+1}/{n_batches}] ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        results.append({
            "batch": i + 1,
            "accuracy": acc,
            "f1": f1,
            "auc": auc
        })

        # ==================================================
        # Wyjaśnialnosc per batch (TOP features)
        # ==================================================
        coef_array = np.asarray(base_model.coef_).reshape(-1)

        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": coef_array
        })

        coef_df["Coefficient"] = pd.to_numeric(coef_df["Coefficient"], errors="coerce")
        coef_df = coef_df.dropna()
        coef_df["abs_coef"] = coef_df["Coefficient"].abs()
        coef_sorted = coef_df.sort_values("abs_coef", ascending=False)

        top_feats = coef_sorted.head(10)

        plt.figure(figsize=(9, 6))
        sns.barplot(x="Coefficient", y="Feature", data=top_feats, orient="h")
        plt.title(f"Top 10 cech – batch {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_features_batch_{i+1}.png"))
        plt.close()

    # ==================================================
    # Save base model & used data
    # ==================================================
    joblib.dump(base_model, base_model_path)
    new_df.to_csv(used_path, index=False)

    # ==================================================
    # Global calibration
    # ==================================================
    if calibration:
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        model.fit(X, y)
        joblib.dump(model, calibrated_model_path)
    else:
        model = base_model

    # ==================================================
    # Global evaluation + global explainability
    # ==================================================
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc_g = accuracy_score(y, y_pred)
    f1_g = f1_score(y, y_pred)
    auc_g = roc_auc_score(y, y_prob)

    print(f"\n[GLOBAL] ACC={acc_g:.4f}, F1={f1_g:.4f}, AUC={auc_g:.4f}")

    # --- Global coefficients ---
    coef_array = np.asarray(base_model.coef_).reshape(-1)
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coef_array
    })
    coef_df["Coefficient"] = pd.to_numeric(coef_df["Coefficient"], errors="coerce")
    coef_df = coef_df.dropna()
    coef_df["abs_coef"] = coef_df["Coefficient"].abs()
    coef_sorted = coef_df.sort_values("abs_coef", ascending=False)

    # Global plots
    for name, data in {
        "top_global": coef_sorted.head(10),
        "top_positive": coef_df.sort_values("Coefficient", ascending=False).head(10),
        "top_negative": coef_df.sort_values("Coefficient").head(10)
    }.items():
        plt.figure(figsize=(9, 6))
        sns.barplot(x="Coefficient", y="Feature", data=data, orient="h")
        plt.title(name.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.close()

    # ==================================================
    # Save metrics
    # ==================================================
    pd.DataFrame(results).to_csv(
        os.path.join(models_dir, "incremental_results.csv"),
        index=False
    )



def run_incremental_training(train_data_path, models_dir="models/log_reg_inc", restart=False):
    """
    Uruchamia incremental learning dla regresji logistycznej.

    Parametry:
    - train_data_path: ścieżka do pełnego zbioru treningowego
    - models_dir: folder, gdzie zapisywane są modele i plik 'train_used.csv'
    - restart: jeśli True, usuwa poprzedni model i dane, ucząc od nowa
    """

    # Ścieżki do plików
    base_model_path = os.path.join(models_dir, "base_model_sgd.joblib")
    calibrated_model_path = os.path.join(models_dir, "calibrated_model.joblib")
    used_path = os.path.join(models_dir, "train_used.csv")

    # 🔹 Restart pełny
    if restart:
        for path in [base_model_path, calibrated_model_path, used_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[INFO] Usunięto {path} – pipeline będzie trenowany od nowa")

    # 🔹 Wywołanie incremental pipeline
    run_logistic_regression_incremental(
        full_train_path=train_data_path,
        batch_size=25,
        models_dir=models_dir,
        output_dir="outputs/log_reg_inc",
        pairwise_poly=True,
        pairwise_poly_top10=False,
        pairwise_linear=False,
        calibration=True
    )

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