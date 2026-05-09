import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("1. Trenowanie OSTATECZNEGO zwycięskiego modelu (R2 = 0.77)...")
df = pd.read_csv('3_finalna_baza_do_modelowania.csv')

# Winsoryzacja
kolumny_delt = [col for col in df.columns if 'Delta' in col]
for col in kolumny_delt:
    dolna_granica = df[col].quantile(0.01)
    gorna_granica = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=dolna_granica, upper=gorna_granica)

df['Target_Forest_Delta_Next_Year'] = df.groupby('Country_Name')['Forest_Area_%_Delta'].shift(-1)
df = df.dropna(subset=['Target_Forest_Delta_Next_Year'])

train_mask = df['Year'] <= 2018
test_mask = df['Year'] > 2018

kolumny_do_odrzucenia = ['Country_Name', 'Year', 'Target_Forest_Delta_Next_Year', 'Forest_Area_%_Delta']
cechy_wejsciowe = [col for col in df.columns if col not in kolumny_do_odrzucenia]

X_train = df.loc[train_mask, cechy_wejsciowe]
y_train = df.loc[train_mask, 'Target_Forest_Delta_Next_Year']
X_test = df.loc[test_mask, cechy_wejsciowe]

model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

print("\n2. Uruchamianie algorytmu SHAP (Explainable AI)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("\n3. Generowanie Wykresu Kierunków (Beeswarm Plot)...")
plt.figure(figsize=(12, 10))
# Wartości na prawo od pionowej linii oznaczają dodawanie lasu. Wartości na lewo to wycinka.
plt.title("Wpływ Globalnych Wskaźników na Przyszłoroczną Dynamikę Zmian Lasu\n(SHAP > 0: Wzrost lasu | SHAP < 0: Wylesianie)", fontsize=14, pad=20)

shap.summary_plot(shap_values, X_test, show=False)

plt.tight_layout()
plt.savefig('Wykres_SHAP_Finalny.png', dpi=300, bbox_inches='tight')
print("✅ Gotowe! Sprawdź folder projektu - powstał plik 'Wykres_SHAP_Finalny.png'.")