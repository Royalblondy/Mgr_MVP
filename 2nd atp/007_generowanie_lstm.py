import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

# Wyłączamy zbędne ostrzeżenia
warnings.filterwarnings('ignore')

print("1. Ładowanie ostatecznej bazy danych do sieci LSTM...")
df = pd.read_csv('4_baza_gotowa_na_ml.csv')

# Definiujemy epoki czasowe
EPOKA_LSTM_TRAIN = 2014
cechy = [col for col in df.columns if '_Lag1' in col]

print("\n2. Skalowanie danych (Unikamy wycieku!)")
maska_treningowa = df['Year'] <= EPOKA_LSTM_TRAIN
scaler = MinMaxScaler()
scaler.fit(df.loc[maska_treningowa, cechy])

df_scaled = df.copy()
df_scaled[cechy] = scaler.transform(df[cechy])

print("\n3. Formatowanie 3D: Tworzenie 3-LETNICH okien czasowych...")
OKNO_CZASOWE = 3  # <--- NASZA ZMIANA EKSPERYMENTALNA

X_seq, y_seq, meta_seq = [], [], []

for kraj, grupa in df_scaled.groupby('Country_Name'):
    grupa = grupa.sort_values('Year').reset_index(drop=True)

    if len(grupa) <= OKNO_CZASOWE:
        continue

    wartosci_cech = grupa[cechy].values
    target = grupa['Forest_Change_%'].values
    lata = grupa['Year'].values
    kraje = grupa['Country_Name'].values

    for i in range(OKNO_CZASOWE, len(grupa)):
        X_seq.append(wartosci_cech[i - OKNO_CZASOWE: i])
        y_seq.append(target[i])
        meta_seq.append((kraje[i], lata[i]))

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
meta_seq = np.array(meta_seq)

print("\n4. Dzielenie okien na zbiór uczący LSTM (do 2014) i resztę...")
train_idx = meta_seq[:, 1].astype(int) <= EPOKA_LSTM_TRAIN

X_train_lstm = X_seq[train_idx]
y_train_lstm = y_seq[train_idx]

print(f" -> Ilość 3-letnich sekwencji treningowych: {len(X_train_lstm)}")

print("\n5. Budowa i Trening Sieci Neuronowej LSTM (Okno 3 lata)...")
model_lstm = Sequential()
model_lstm.add(LSTM(32, input_shape=(OKNO_CZASOWE, len(cechy)), activation='relu'))
model_lstm.add(Dense(16, activation='relu'))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=1)

print("\n6. Generowanie prognoz LSTM dla WSZYSTKICH lat...")
wszystkie_predykcje = model_lstm.predict(X_seq)

print("\n7. Wzbogacenie oryginalnej bazy danych o predykcje LSTM...")
df_predykcje = pd.DataFrame({
    'Country_Name': meta_seq[:, 0],
    'Year': meta_seq[:, 1].astype(int),
    'Predykcja_LSTM': wszystkie_predykcje.flatten()
})

df_hybryda = pd.merge(df, df_predykcje, on=['Country_Name', 'Year'], how='inner')

# Zapisujemy pod nową nazwą!
df_hybryda.to_csv('5_baza_z_lstm_3lata.csv', index=False)
print("=" * 60)
print(" ARCHITEKTURA SOTA GOTOWA! Plik '5_baza_z_lstm_3lata.csv' zapisany.")
print("=" * 60)