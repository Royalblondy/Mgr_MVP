import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1. Ładowanie zhybrydyzowanej bazy danych (z ukrytym zmysłem LSTM)...")
df = pd.read_csv('4_baza_z_lstm_sota.csv')

print("2. Definiowanie zbioru uczącego i testowego (Podział 2018)...")
train_mask = df['Year'] <= 2018
test_mask = df['Year'] > 2018

# Definiujemy cechy. Nasz nowy nabytek 'LSTM_Sygnal_Trendu' wchodzi do gry!
kolumny_do_odrzucenia = [
    'Country_Name', 'Year', 'Target_Forest_Delta_Next_Year',
    'Forest_Area_%_Delta'
]
cechy_wejsciowe = [col for col in df.columns if col not in kolumny_do_odrzucenia]

X_train = df.loc[train_mask, cechy_wejsciowe]
y_train = df.loc[train_mask, 'Target_Forest_Delta_Next_Year']

X_test = df.loc[test_mask, cechy_wejsciowe]
y_test = df.loc[test_mask, 'Target_Forest_Delta_Next_Year']

print(f" -> Zbiór Treningowy: {len(X_train)} przypadków.")
print(f" -> Zbiór Testowy: {len(X_test)} przypadków.")

print("\n3. Budowa i Trening Modelu XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
# print("\n3. Budowa i Trening Modelu XGBoost (Zmuszamy do pracy!)...")
# model = xgb.XGBRegressor(
#     n_estimators=1000,       # Znacznie więcej drzew do zbudowania
#     learning_rate=0.01,      # Uczymy się 3x wolniej i dokładniej
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.4,    # Brutalne zasłanianie danych: ukrywamy LSTM w 60% przypadków!
#     reg_lambda=5.0,          # Kara L2 za opieranie się na jednej cesze (domyślnie jest 1)
#     reg_alpha=1.0,           # Kara L1 promująca różnorodność cech
#     random_state=42
# )
#
# model.fit(X_train, y_train)

print("\n4. Egzamin na latach > 2018...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("  WYNIKI OSTATECZNEGO MODELU HYBRYDOWEGO (XGBoost + LSTM)")
print("="*60)
print(f"R² (Współczynnik determinacji): {r2:.4f} (Do pobicia: 0.7762)")
print(f"MAE (Średni błąd bezwzględny):   {mae:.5f}")
print(f"RMSE (Błąd średniokwadratowy):   {rmse:.5f}")
print("="*60)

print("\n5. Tabela Ważności Cech - Gdzie uplasowało się LSTM?")
waznosc = pd.DataFrame({
    'Cecha': cechy_wejsciowe,
    'Waznosc_%': model.feature_importances_ * 100
}).sort_values(by='Waznosc_%', ascending=False)

print(waznosc.to_string(index=False))