import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("1. Trenowanie zwycięskiego modelu XGBoost...")
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

model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

print("2. Uruchamianie silnika SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Nowsze wersje SHAP (dla wykresu Waterfall) potrzebują specjalnego obiektu:
shap_obj = explainer(X_test)

print("3. Generowanie Galerii Wykresów...\n")

# --- WYKRES 1: Klasyczny Słupkowy (Czysta ważność bez kierunków) ---
print(" -> Rysowanie: 1_SHAP_Slupkowy.png")
plt.figure(figsize=(10, 8))
plt.title("Globalna Ważność Cech (Absolute SHAP value)", fontsize=14)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('1_SHAP_Slupkowy.png', dpi=300, bbox_inches='tight')
plt.close()

# --- WYKRES 2: Wykres Zależności (Szukamy Środowiskowej Krzywej Kuznetsa dla PKB) ---
print(" -> Rysowanie: 2_SHAP_Zaleznosc_PKB.png")
plt.figure(figsize=(8, 6))
plt.title("Jak PKB per Capita wpływa na lasy? (Poszukiwanie punktu zwrotnego)", fontsize=12)
shap.dependence_plot("GDP_per_Capita", shap_values, X_test, interaction_index=None, show=False)
plt.tight_layout()
plt.savefig('2_SHAP_Zaleznosc_PKB.png', dpi=300, bbox_inches='tight')
plt.close()

# --- WYKRES 3: Kaskada dla jednego kraju (Local Explainability) ---
# Bierzemy pierwszy wiersz ze zbioru testowego (indeks 0)
print(" -> Rysowanie: 3_SHAP_Kaskada_Lokalna.png")
plt.figure(figsize=(10, 6))
plt.title("Analiza Mikroskopowa: Dlaczego model podjął taką decyzję dla Kraju X?", fontsize=12, pad=20)
shap.plots.waterfall(shap_obj[0], show=False)
plt.tight_layout()
plt.savefig('3_SHAP_Kaskada_Lokalna.png', dpi=300, bbox_inches='tight')
plt.close()

# --- WYKRES 4: Interakcja (Krowy + Prąd) ---
print(" -> Rysowanie: 4_SHAP_Interakcja.png")
plt.figure(figsize=(8, 6))
plt.title("Efekt mnożnikowy: Powierzchnia Pastwisk a Dostęp do Prądu", fontsize=12)
# Sprawdzamy jak Pastwiska wpływają na las, z uwzględnieniem (kolorem) dostępu do prądu
shap.dependence_plot("Pasture_Area", shap_values, X_test, interaction_index="Rural_Clean_Fuel_%", show=False)
plt.tight_layout()
plt.savefig('4_SHAP_Interakcja.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Wernisaż zakończony! 4 obrazy czekają w Twoim folderze.")