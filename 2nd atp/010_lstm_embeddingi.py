import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import warnings

warnings.filterwarnings('ignore')

print("1. Ładowanie i skalowanie bazy danych...")
df = pd.read_csv('4_baza_gotowa_na_ml.csv')
EPOKA_LSTM_TRAIN = 2014
OKNO_CZASOWE = 3

cechy = [col for col in df.columns if '_Lag1' in col]

maska_treningowa = df['Year'] <= EPOKA_LSTM_TRAIN
scaler = MinMaxScaler()
scaler.fit(df.loc[maska_treningowa, cechy])

df_scaled = df.copy()
df_scaled[cechy] = scaler.transform(df[cechy])

print("\n2. Tworzenie 3-letnich okien czasowych dla LSTM...")
X_seq, y_seq, meta_seq = [], [], []

for kraj, grupa in df_scaled.groupby('Country_Name'):
    grupa = grupa.sort_values('Year').reset_index(drop=True)
    if len(grupa) <= OKNO_CZASOWE: continue

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

train_idx = meta_seq[:, 1].astype(int) <= EPOKA_LSTM_TRAIN
X_train_lstm, y_train_lstm = X_seq[train_idx], y_seq[train_idx]

print("\n3. Budowa Sieci Neuronowej z warstwą EMBEDDINGOWĄ...")
# ARCHITEKTURA SOTA:
model_lstm = Sequential()
model_lstm.add(Input(shape=(OKNO_CZASOWE, len(cechy))))
model_lstm.add(LSTM(32, activation='relu'))
# TO JEST NASZA MAGICZNA WARSTWA - Zmuszamy sieć do streszczenia wiedzy w 4 liczbach
model_lstm.add(Dense(4, activation='relu', name='warstwa_ekstrakcji'))
model_lstm.add(Dense(1))  # Ostateczna predykcja tylko po to, żeby móc trenować błąd

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
print(" -> Trening LSTM zakończony.")

print("\n4. Amputacja sieci - wyciągamy wiedzę zamiast predykcji!")
# "Odcinamy" ostatnią warstwę. Zostawiamy tylko tę, która generuje 4 cechy.
ekstraktor = Model(inputs=model_lstm.inputs, outputs=model_lstm.get_layer('warstwa_ekstrakcji').output)

# Przekształcamy całą historię świata na 4 nowe kolumny ukryte
ukryte_cechy_lstm = ekstraktor.predict(X_seq)

print("\n5. Łączenie wymiarów (Klasyczne ML + Deep Learning)...")
df_emb = pd.DataFrame(ukryte_cechy_lstm, columns=['LSTM_Zmysl_1', 'LSTM_Zmysl_2', 'LSTM_Zmysl_3', 'LSTM_Zmysl_4'])
df_meta = pd.DataFrame({'Country_Name': meta_seq[:, 0], 'Year': meta_seq[:, 1].astype(int)})
df_nowe_cechy = pd.concat([df_meta, df_emb], axis=1)

# Fuzja ostateczna
df_ostateczne = pd.merge(df, df_nowe_cechy, on=['Country_Name', 'Year'], how='inner')
df_ostateczne['Country_Name'] = df_ostateczne['Country_Name'].astype('category')

print("\n6. Trening ostatecznego XGBoosta ze wzmocnieniem...")
train_mask = df_ostateczne['Year'] <= 2018
test_mask = df_ostateczne['Year'] > 2018

cechy_wejsciowe = [col for col in df_ostateczne.columns if col not in ['Year', 'Forest_Change_%']]

X_train = df_ostateczne.loc[train_mask, cechy_wejsciowe]
y_train = df_ostateczne.loc[train_mask, 'Forest_Change_%']
X_test = df_ostateczne.loc[test_mask, cechy_wejsciowe]
y_test = df_ostateczne.loc[test_mask, 'Forest_Change_%']

model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, enable_categorical=True,
                             random_state=42)
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f" 🚀 WYNIKI MODELU Z EMBEDDINGAMI LSTM: R² = {r2:.4f}")
print("=" * 60)

waznosc = pd.DataFrame({'Cecha': cechy_wejsciowe, 'Waznosc_%': model_xgb.feature_importances_ * 100}).sort_values(
    by='Waznosc_%', ascending=False)
print("\nTop Cechy (Czy XGBoost użył w końcu LSTM?):")
print(waznosc.head(10).to_string(index=False))