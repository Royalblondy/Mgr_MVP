import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Ładowanie i czyszczenie danych...")
df = pd.read_csv('mvp_features_v2.csv')

czarna_lista = [
    'World', 'Africa', 'Eastern Africa', 'Middle Africa', 'Northern Africa',
    'Southern Africa', 'Western Africa', 'Americas', 'Northern America',
    'Central America', 'Caribbean', 'South America', 'Asia', 'Central Asia',
    'Eastern Asia', 'Southern Asia', 'South-eastern Asia', 'Western Asia',
    'Europe', 'Eastern Europe', 'Northern Europe', 'Southern Europe',
    'Western Europe', 'Oceania', 'Australia and New Zealand', 'Melanesia',
    'Micronesia', 'Polynesia', 'European Union (27)', 'European Union (28)',
    'Least Developed Countries', 'Land Locked Developing Countries (LLDCs)',
    'Small Island Developing States', 'Low Income Food Deficit Countries',
    'Net Food Importing Developing Countries', 'Annex I countries', 'Non-Annex I countries', 'OECD',
    'Bahrain', 'Malta', 'Iceland'
]

df_clean = df[~df['Area'].isin(czarna_lista)].copy()
df_clean['Forest_Area_Lag1'] = df_clean.groupby('Area')['Forest_Area'].shift(1)
df_clean = df_clean.dropna()

# 1. PRZYGOTOWANIE CECH (Bez tekstowego 'Area')
features = ['Forest_Area_Lag1', 'Agri_Area_Lag1', 'Cattle_Head_Lag1', 'Cattle_Change_Lag1', 'Cattle_Density_Lag1']
target = 'Forest_Change_%'

train_df = df_clean[df_clean['Year'] <= 2015].copy()
test_df  = df_clean[df_clean['Year'] >= 2020].copy()

# 2. NORMALIZACJA (Skalowanie do 0-1) - KRYTYCZNE DLA SIECI NEURONOWYCH!
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Skalujemy cechy i cel osobno
X_train_scaled = scaler_X.fit_transform(train_df[features])
y_train_scaled = scaler_y.fit_transform(train_df[[target]])

X_test_scaled = scaler_X.transform(test_df[features])
y_test_scaled = scaler_y.transform(test_df[[target]])

# 3. FORMATOWANIE POD LSTM [próbki, kroki_czasowe, cechy]
# Robimy proste ujęcie: 1 krok czasowy, w którym są zawarte nasze opóźnienia (Lags)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

print("Budowa prostej sieci LSTM...")
# 4. BUDOWA ARCHITEKTURY (Bez fajerwerków)
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(16, activation='relu'))
model.add(Dense(1)) # Wyjście - nasza jedna zgadywana liczba

model.compile(optimizer='adam', loss='mse')

# 5. TRENOWANIE SIECI
print("Trenowanie sieci (to może chwilę potrwać)...")
model.fit(X_train_lstm, y_train_scaled, epochs=20, batch_size=32, verbose=1)

# 6. PRZEWIDYWANIE I ODSKALOWANIE WYNIKÓW
y_pred_scaled = model.predict(X_test_lstm)
# Musimy przywrócić wyniki do normalnych procentów, żeby można było je porównać z XGBoostem
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 7. METRYKI (Takie same jak w XGBoost, do równego porównania)
mae = mean_absolute_error(test_df[target], y_pred)
rmse = np.sqrt(mean_squared_error(test_df[target], y_pred))
r2 = r2_score(test_df[target], y_pred)

print("\n--- WYNIKI SIECI LSTM ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")