import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1. Ładowanie ostatecznej macierzy danych...")
df = pd.read_csv('4_baza_gotowa_na_ml.csv')

# Zamieniamy nazwę kraju na typ "Category", żeby XGBoost wiedział, że to tekst, a nie liczba
df['Country_Name'] = df['Country_Name'].astype('category')

print("2. Podział na zbiór Treningowy (Przeszłość) i Testowy (Przyszłość)...")
# Uczymy model na latach do 2018 włącznie
train_mask = df['Year'] <= 2018
test_mask = df['Year'] > 2018

# Definiujemy, co jest wejściem (X), a co wyjściem (y)
cechy_wejsciowe = [col for col in df.columns if col not in ['Year', 'Forest_Change_%']]
X_train = df.loc[train_mask, cechy_wejsciowe]
y_train = df.loc[train_mask, 'Forest_Change_%']

X_test = df.loc[test_mask, cechy_wejsciowe]
y_test = df.loc[test_mask, 'Forest_Change_%']

print(f" -> Uczymy się na {len(X_train)} przypadkach z lat 2000-2018.")
print(f" -> Testujemy na {len(X_test)} przypadkach z lat 2019-2022.")

print("\n3. Trening Sztucznej Inteligencji (XGBoost)...")
# Budujemy i konfigurujemy "mózg" naszego algorytmu
model = xgb.XGBRegressor(
    n_estimators=300,        # Ilość drzew decyzyjnych
    learning_rate=0.05,      # Szybkość uczenia (im mniejsza, tym model ostrożniejszy)
    max_depth=6,             # Maksymalna głębokość "myślenia" drzewa
    subsample=0.8,           # Losuje 80% danych dla każdego drzewa (zapobiega przeuczeniu)
    enable_categorical=True, # Pozwala modelowi czytać nazwy krajów!
    random_state=42
)

model.fit(X_train, y_train)

print("\n4. Egzamin: Przewidywanie lat 2019-2022...")
y_pred = model.predict(X_test)

# Ocena modelu matematycznie
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(" 🏆 WYNIKI MODELU XGBOOST NA NIEZNANYCH DANYCH")
print("="*50)
print(f"R² (Współczynnik determinacji): {r2:.4f} (Idealny wynik = 1.0)")
print(f"MAE (Średni błąd bezwzględny):   {mae:.4f} p.p. (Mylimy się średnio o tyle %)")
print(f"RMSE (Błąd średniokwadratowy):   {rmse:.4f} p.p.")
print("="*50)

print("\n5. Co było dla modelu najważniejsze w podejmowaniu decyzji?")
waznosc = pd.DataFrame({
    'Cecha': cechy_wejsciowe,
    'Waznosc_%': model.feature_importances_ * 100
}).sort_values(by='Waznosc_%', ascending=False)

print(waznosc.to_string(index=False))