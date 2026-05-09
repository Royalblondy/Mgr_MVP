import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Ładowanie i czyszczenie (standardowa procedura)
df = pd.read_csv('mvp_features_v2.csv')
czarna_lista = ['World', 'Africa', 'Europe', 'Americas', 'Asia', 'Oceania', 'European Union (27)', 'Bahrain', 'Malta',
                'Iceland']  # skrócona dla czytelności kodu
df_clean = df[~df['Area'].isin(czarna_lista)].copy()

features = ['Forest_Area', 'Agri_Area', 'Cattle_Head', 'Cattle_Change_%', 'Cattle_Density']
target = 'Forest_Change_%'

# 2. Skalowanie danych przed robieniem sekwencji
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df_clean[features] = scaler_X.fit_transform(df_clean[features])
df_clean[[target]] = scaler_y.fit_transform(df_clean[[target]])


# 3. FUNKCJA TWORZĄCA SEKWENCJE (3 lata -> 4. rok)
def create_sequences(data, window_size=3):
    X, y, years, areas = [], [], [], []
    for area in data['Area'].unique():
        area_df = data[data['Area'] == area].sort_values('Year')
        values = area_df[features].values
        targets = area_df[target].values
        year_vals = area_df['Year'].values

        for i in range(window_size, len(area_df)):
            X.append(values[i - window_size:i])  # weź 3 poprzednie lata
            y.append(targets[i])  # przewiduj obecny rok
            years.append(year_vals[i])
            areas.append(area)

    return np.array(X), np.array(y), np.array(years), np.array(areas)


X_all, y_all, years_all, areas_all = create_sequences(df_clean, window_size=3)

# 4. Podział czasowy na gotowych sekwencjach
train_mask = years_all <= 2015
test_mask = years_all >= 2020

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test, y_test = X_all[test_mask], y_all[test_mask]

print(f"Rozmiar danych treningowych: {X_train.shape}")  # Powinno być (N, 3, 5)

# 5. Budowa modelu LSTM
model = Sequential([
    LSTM(64, activation='relu', input_shape=(3, 5), return_sequences=False),
    Dropout(0.2),  # mały "bezpiecznik" przed overfittingiem
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 6. Trenowanie
print("Trenowanie LSTM z pamięcią 3-letnią...")
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# 7. Wyniki
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- WYNIKI LSTM (3 LATA PAMIĘCI) ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")