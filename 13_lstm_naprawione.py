import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Ładowanie danych...")
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

# --- MAGIA: ONE-HOT ENCODING (Uczymy sieć geografii) ---
print("Tłumaczenie geografii na język macierzy (One-Hot Encoding)...")
# Tworzymy kopię kolumny Area do grupowania, a oryginalną zamieniamy na jedynki i zera
df_encoded = pd.get_dummies(df_clean, columns=['Area'], dtype=float)
df_encoded['Area_Name'] = df_clean['Area']

# Definiujemy cechy (wszystko oprócz roku, targetu i nazw)
cechy_do_usuniecia = ['Year', 'Forest_Change_%', 'Area_Name', 'Risk_Class']
features = [col for col in df_encoded.columns if col not in cechy_do_usuniecia]
target = 'Forest_Change_%'

# Skalowanie
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df_encoded[features] = scaler_X.fit_transform(df_encoded[features])
df_encoded[[target]] = scaler_y.fit_transform(df_encoded[[target]])


# Funkcja tworząca sekwencje
def create_sequences(data, window_size=3):
    X, y, years = [], [], []
    for area in data['Area_Name'].unique():
        area_df = data[data['Area_Name'] == area].sort_values('Year')
        values = area_df[features].values
        targets = area_df[target].values
        year_vals = area_df['Year'].values

        for i in range(window_size, len(area_df)):
            X.append(values[i - window_size:i])
            y.append(targets[i])
            years.append(year_vals[i])

    return np.array(X), np.array(y), np.array(years)


print("Budowanie sekwencji 3-letnich...")
X_all, y_all, years_all = create_sequences(df_encoded, window_size=3)

# Podział czasowy (Time-Based Split)
train_mask = years_all <= 2015
test_mask = years_all >= 2020

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test, y_test = X_all[test_mask], y_all[test_mask]

# Budowa modelu LSTM z poprawionym kształtem wejścia
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Zabezpieczenie przed przeuczeniem (Early Stopping)
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

print(f"Trenowanie na danych z kontekstem... (ilość cech: {X_train.shape[2]})")
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1, callbacks=[early_stop])

# Wyniki
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- NAPRAWIONE WYNIKI LSTM (Z GEOGRAFIĄ) ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")