import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import warnings

warnings.filterwarnings('ignore')

print("1. Ładowanie potężnej bazy danych...")
df = pd.read_csv('3_finalna_baza_do_modelowania.csv')

print("2. Tarcza Inżynierska (Winsoryzacja)...")
# Musimy oczyścić anomalię 6400% również dla LSTM, inaczej sieć neuronowa zwariuje
kolumny_delt = [col for col in df.columns if 'Delta' in col]
for col in kolumny_delt:
    dolna_granica = df[col].quantile(0.01)
    gorna_granica = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=dolna_granica, upper=gorna_granica)

print("3. Inżynieria Czasu: Ustawianie Targetu...")
df['Target_Forest_Delta_Next_Year'] = df.groupby('Country_Name')['Forest_Area_%_Delta'].shift(-1)
df = df.dropna(subset=['Target_Forest_Delta_Next_Year'])

kolumny_do_odrzucenia = ['Country_Name', 'Year', 'Target_Forest_Delta_Next_Year', 'Forest_Area_%_Delta']
cechy_wejsciowe = [col for col in df.columns if col not in kolumny_do_odrzucenia]

print("4. Restrykcyjny podział czasu (Ochrona przed Data Leakage!)...")
EPOKA_LSTM_TRAIN = 2014
maska_treningowa = df['Year'] <= EPOKA_LSTM_TRAIN

scaler = MinMaxScaler()
# Skaler uczy się rozkładu świata TYLKO do 2014 roku
scaler.fit(df.loc[maska_treningowa, cechy_wejsciowe])

df_scaled = df.copy()
df_scaled[cechy_wejsciowe] = scaler.transform(df[cechy_wejsciowe])

print("5. Cięcie na 3-letnie sekwencje czasowe...")
OKNO_CZASOWE = 3
X_seq, y_seq, meta_seq = [], [], []

for kraj, grupa in df_scaled.groupby('Country_Name'):
    grupa = grupa.sort_values('Year').reset_index(drop=True)
    if len(grupa) <= OKNO_CZASOWE:
        continue

    wartosci_cech = grupa[cechy_wejsciowe].values
    target = grupa['Target_Forest_Delta_Next_Year'].values
    lata = grupa['Year'].values
    kraje = grupa['Country_Name'].values

    for i in range(OKNO_CZASOWE, len(grupa)):
        X_seq.append(wartosci_cech[i - OKNO_CZASOWE: i])
        y_seq.append(target[i])
        meta_seq.append((kraje[i], lata[i]))

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
meta_seq = np.array(meta_seq)

train_idx = meta_seq[:, 1].astype(int) <= EPOKA_LSTM_TRAIN
X_train_lstm = X_seq[train_idx]
y_train_lstm = y_seq[train_idx]

print(f" -> Gotowe. Trenujemy na {len(X_train_lstm)} sekwencjach.")

print("\n6. Budowa i Trening Sieci Neuronowej LSTM...")
# Używam czystego formatu Input(), by uniknąć starych ostrzeżeń Keras
model_lstm = Sequential([
    Input(shape=(OKNO_CZASOWE, len(cechy_wejsciowe))),
    LSTM(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=32, verbose=1)

print("\n7. Generowanie 'Szóstego Zmysłu' dla wszystkich lat...")
# Sieć wypluwa diagnozę. Dla lat po 2014 robi to w 100% z zasłoniętymi oczami!
wszystkie_predykcje = model_lstm.predict(X_seq)

df_predykcje = pd.DataFrame({
    'Country_Name': meta_seq[:, 0],
    'Year': meta_seq[:, 1].astype(int),
    'LSTM_Sygnal_Trendu': wszystkie_predykcje.flatten()
})

print("\n8. Zapisywanie zhybrydyzowanej bazy danych...")
df_ostateczne = pd.merge(df, df_predykcje, on=['Country_Name', 'Year'], how='inner')
df_ostateczne.to_csv('4_baza_z_lstm_sota.csv', index=False)
print("=" * 60)
print(" HYBRYDA GOTOWA! Zapisano '4_baza_z_lstm_sota.csv'.")
print("=" * 60)