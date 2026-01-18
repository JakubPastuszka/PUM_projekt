#!/bin/bash

# Jeśli nie podasz argumentu, skrypt użyje obecnego folderu "."
TARGET_DIR=${1:-"."}

echo "🚀 Rozpoczynam konfigurację struktury w: $TARGET_DIR"

# 1. Tworzenie katalogów (używając -p dla bezpieczeństwa)
mkdir -p "$TARGET_DIR/data/raw"
mkdir -p "$TARGET_DIR/data/processed"
mkdir -p "$TARGET_DIR/notebooks"
mkdir -p "$TARGET_DIR/src"
mkdir -p "$TARGET_DIR/models"
mkdir -p "$TARGET_DIR/reports/figures"
mkdir -p "$TARGET_DIR/outputs"

# 2. Tworzenie plików .gitkeep
touch "$TARGET_DIR/data/.gitkeep"
touch "$TARGET_DIR/models/.gitkeep"
touch "$TARGET_DIR/outputs/.gitkeep"

# 3. Tworzenie szkieletu kodu
touch "$TARGET_DIR/src/__init__.py"
touch "$TARGET_DIR/src/data_processing.py"
touch "$TARGET_DIR/src/features.py"
touch "$TARGET_DIR/src/visualization.py"

# 4. Generowanie requirements.txt (z Scikit-learn i SHAP dla wyjaśnialności) [cite: 12, 17]
cat <<EOL > "$TARGET_DIR/requirements.txt"
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
jupyter
notebook
EOL

echo "Struktura gotowa! Pamiętaj, aby umieścić plik CSV w data/raw/."