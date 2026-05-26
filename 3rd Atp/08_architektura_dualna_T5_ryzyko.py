import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# KROK 1: FEATURE ENGINEERING - SKUMULOWANY TARGET (T+5)
# ==============================================================================
def create_cumulative_target(df, target_col='Delta_Forest', horizon=5):
    print(f"1. Generowanie skumulowanego targetu dla horyzontu {horizon} lat...")
    df = df.sort_values(by=['Year', 'Country_ID'])

    df['Temp_Target_T1'] = df.groupby('Country_ID')[target_col].shift(-1)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    df['Target_Cumulative_T5'] = df.groupby('Country_ID')['Temp_Target_T1'] \
                                   .rolling(window=indexer, min_periods=horizon) \
                                   .sum() \
                                   .reset_index(level=0, drop=True)

    df = df.drop(columns=['Temp_Target_T1'])
    return df

# ==============================================================================
# KROK 2: WIELOKRYTERIALNOŚĆ - KLASYFIKACJA (TOLERANCJA NA UŁAMKI)
# ==============================================================================
def assign_risk_class(val):
    """
    0 - Niskie ryzyko / Zalesianie (> +1%)
    1 - Średnie ryzyko / Stabilność (-1% do +1%)
    2 - Wysokie ryzyko / Wylesianie (< -1%)
    """
    if pd.isna(val):
        return np.nan
    elif val < -0.01:
        return int(2)
    elif val <= 0.01:
        return int(1)
    else:
        return int(0)

# ==============================================================================
# KROK 3: RYGORYSTYCZNY PODZIAŁ Z GAPEM (DATA LEAKAGE PREVENTION)
# ==============================================================================
def train_test_split_panel_with_gap(df, test_start_year, gap=5):
    print(f"2. Podział na zbiory (Train/Test) z ochroną przed Data Leakage (Gap={gap} lat)...")
    train_end_year = test_start_year - gap - 1

    df_clean = df.dropna(subset=['Target_Cumulative_T5', 'Risk_Class'])

    train_df = df_clean[df_clean['Year'] <= train_end_year]
    test_df = df_clean[df_clean['Year'] >= test_start_year]

    print(f"   -> Zbiór Treningowy: do roku {train_end_year} włącznie.")
    print(f"   -> Zbiór Testowy: od roku {test_start_year}.")

    return train_df, test_df

# ==============================================================================
# GŁÓWNY PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("WCZYTYWANIE PEŁNEJ BAZY DANYCH...")
    # Wczytanie prawdziwej bazy
    df = pd.read_csv('3_finalna_baza_do_modelowania.csv')

    # Mapowanie nazw kolumn na te używane w logice funkcji
    df = df.rename(columns={
        'Country_Name': 'Country_ID',
        'Forest_Area_%_Delta': 'Delta_Forest'
    })

    # Inżynieria Cech (wyliczanie celów T+5)
    df = create_cumulative_target(df, horizon=5)
    df['Risk_Class'] = df['Target_Cumulative_T5'].apply(assign_risk_class)

    # Definicja cech wejściowych - ODCINAMY WSZYSTKO CO JEST CELEM LUB IDENTYFIKATOREM!
    kolumny_do_odrzucenia = [
        'Country_ID',
        'Year',
        'Delta_Forest',           # To jest T+1 target, nie może wejść jako zmienna
        'Target_Cumulative_T5',   # Nasz nowy target ilościowy
        'Risk_Class'              # Nasz nowy target jakościowy
    ]
    features = [col for col in df.columns if col not in kolumny_do_odrzucenia]
    print(f"Użyte cechy ({len(features)}): {features}")

    # Podział na zbiory (Ustawiamy test od 2015, żeby wyrobić 5-letnie okno do 2022)
    train_df, test_df = train_test_split_panel_with_gap(df, test_start_year=2015, gap=5)

    X_train = train_df[features]
    y_train_reg = train_df['Target_Cumulative_T5']
    y_train_clf = train_df['Risk_Class'].astype(int)

    X_test = test_df[features]
    y_test_reg = test_df['Target_Cumulative_T5']
    y_test_clf = test_df['Risk_Class'].astype(int)

    print("\n--- ROZKŁAD KLAS RYZYKA W ZBIORZE TRENINGOWYM ---")
    print(y_train_clf.value_counts())
    print("--------------------------------------------------")

    # ==========================================================================
    # KROK 4: ARCHITEKTURA DUALNA (REGRESJA + KLASYFIKACJA)
    # ==========================================================================
    print("\n3. Inicjalizacja modeli i Grid Search (TimeSeriesSplit)...")
    tscv = TimeSeriesSplit(n_splits=3)

    # ------------------
    # MODEL A (Ilościowy)
    # ------------------
    model_a = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    param_grid_a = {
        'max_depth': [3, 5],
        'learning_rate': [0.03, 0.05],
        'n_estimators': [200, 400]
    }
    grid_a = GridSearchCV(model_a, param_grid_a, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_a.fit(X_train, y_train_reg)
    best_model_a = grid_a.best_estimator_

    pred_a = best_model_a.predict(X_test)
    r2_a = r2_score(y_test_reg, pred_a)
    mae_a = mean_absolute_error(y_test_reg, pred_a)
    rmse_a = np.sqrt(mean_squared_error(y_test_reg, pred_a))

    print("\n[Model A - Ilościowy] Pełne Metryki (T+5):")
    print(f" -> R² (Determinacja): {r2_a:.4f}")
    print(f" -> MAE: {mae_a:.4f}")
    print(f" -> RMSE: {rmse_a:.4f}")

    # ------------------
    # MODEL B (Jakościowy)
    # ------------------
    model_b = xgb.XGBClassifier(random_state=42, n_jobs=-1, objective='multi:softmax', num_class=3)
    param_grid_b = {
        'max_depth': [3, 5],
        'learning_rate': [0.03, 0.05],
        'n_estimators': [200, 400]
    }
    grid_b = GridSearchCV(model_b, param_grid_b, cv=tscv, scoring='f1_macro', n_jobs=-1)
    grid_b.fit(X_train, y_train_clf)
    best_model_b = grid_b.best_estimator_

    pred_b = best_model_b.predict(X_test)
    print(f"\n[Model B - Klasyfikacja Ryzyka] Raport:\n{classification_report(y_test_clf, pred_b)}")

    # Generowanie Macierzy Konfuzji
    cm = confusion_matrix(y_test_clf, pred_b)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Zalesianie (0)', 'Stabilność (1)', 'Wylesianie (2)'],
                yticklabels=['Zalesianie (0)', 'Stabilność (1)', 'Wylesianie (2)'])
    plt.ylabel('Prawdziwa Klasa')
    plt.xlabel('Przewidziana Klasa')
    plt.title('Macierz Konfuzji - Ryzyko T+5')
    plt.tight_layout()
    plt.savefig('Confusion_Matrix_T5.png', dpi=300)
    plt.close()

    # ==========================================================================
    # KROK 5: WYJAŚNIALNE AI (SHAP)
    # ==========================================================================
    print("\n4. Generowanie Wyjaśnień SHAP (dla Modelu A)...")

    explainer = shap.TreeExplainer(best_model_a)
    shap_values = explainer.shap_values(X_test)

    # Wykres 1: Wpływ globalny
    plt.figure(figsize=(10, 6))
    plt.title("Ważność Zmiennych w ujęciu 5-letnim (SHAP Summary)", fontsize=14)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('SHAP_Summary_T5.png', dpi=300)
    plt.close()

    # Wykres 2: Dependence Plot dla Pastwisk
    plt.figure(figsize=(8, 5))
    shap.dependence_plot('Pasture_Area', shap_values, X_test, interaction_index=None, show=False)
    plt.title("Zależność: Pastwiska vs Długoterminowe Wylesianie")
    plt.tight_layout()
    plt.savefig('SHAP_Dependence_Pasture_T5.png', dpi=300)
    plt.close()

    # Wykres 3: Dependence Plot dla Czystej Energii
    plt.figure(figsize=(8, 5))
    shap.dependence_plot('Rural_Clean_Fuel_%', shap_values, X_test, interaction_index=None, show=False)
    plt.title("Zależność: Czysta Energia vs Długoterminowe Wylesianie")
    plt.tight_layout()
    plt.savefig('SHAP_Dependence_CleanFuel_T5.png', dpi=300)
    plt.close()

    print(" Skrypt zakończył działanie. Wygenerowano wykresy SHAP oraz Macierz Konfuzji dla horyzontu 5-letniego.")