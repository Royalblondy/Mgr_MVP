import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1. Ładowanie ZHYBRYDYZOWANEJ macierzy danych (z kolumną LSTM)...")
# Wczytujemy plik wygenerowany przez sieć neuronową!
df = pd.read_csv('5_baza_z_lstm_3lata.csv')

# Zamieniamy nazwę kraju na typ "Category"
df['Country_Name'] = df['Country_Name'].astype('category')

print("2. Podział na zbiór Treningowy i Testowy...")
train_mask = df['Year'] <= 2018
test_mask = df['Year'] > 2018

# Cechy wejściowe - teraz automatycznie zawierają 'Predykcja_LSTM'!
cechy_wejsciowe = [col for col in df.columns if col not in ['Year', 'Forest_Change_%']]

X_train = df.loc[train_mask, cechy_wejsciowe]
y_train = df.loc[train_mask, 'Forest_Change_%']

X_test = df.loc[test_mask, cechy_wejsciowe]
y_test = df.loc[test_mask, 'Forest_Change_%']

print(f" -> Uczymy się na {len(X_train)} przypadkach (od 2005 do 2018).")
print(f" -> Testujemy na {len(X_test)} przypadkach z lat 2019-2022.")

print("\n3. Trening Hybrydowej Sztucznej Inteligencji (XGBoost + LSTM)...")
# Używamy DOKŁADNIE tych samych parametrów co w skrypcie 006, żeby test był sprawiedliwy
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    enable_categorical=True,
    random_state=42
)

model.fit(X_train, y_train)

print("\n4. Egzamin: Przewidywanie lat 2019-2022...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("  WYNIKI OSTATECZNEGO MODELU HYBRYDOWEGO (XGBoost + LSTM)")
print("="*60)
print(f"R² (Współczynnik determinacji): {r2:.4f} (Poprzednio: 0.3144)")
print(f"MAE (Średni błąd bezwzględny):   {mae:.4f} p.p. (Poprzednio: 0.0391)")
print(f"RMSE (Błąd średniokwadratowy):   {rmse:.4f} p.p. (Poprzednio: 0.2433)")
print("="*60)

print("\n5. Tabela Ważności Cech - Gdzie uplasowało się LSTM?")
waznosc = pd.DataFrame({
    'Cecha': cechy_wejsciowe,
    'Waznosc_%': model.feature_importances_ * 100
}).sort_values(by='Waznosc_%', ascending=False)

print(waznosc.to_string(index=False))