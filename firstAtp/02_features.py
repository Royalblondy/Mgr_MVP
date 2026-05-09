import pandas as pd
import numpy as np

print("Ładowanie surowych danych...")
df = pd.read_csv('mvp_data_global.csv')

# 1. Sortujemy dane, żeby upewnić się, że lata idą po kolei dla każdego kraju
df = df.sort_values(by=['Area', 'Year'])

print("Liczenie procentowej zmiany zalesienia...")
# 2. Obliczamy % zmiany lasów w stosunku do poprzedniego roku
df['Forest_Change_%'] = df.groupby('Area')['Forest_Area'].pct_change() * 100

print("Przypisywanie klas ryzyka...")
# 3. Klasy ryzyka dla modelu klasyfikacyjnego (XGBoost)
# Kategoria 2: Spadek zalesienia (Ryzyko!)
# Kategoria 1: Stabilnie (zmiany między -0.1% a 0.1%)
# Kategoria 0: Wzrost zalesienia (Bezpiecznie)
warunki = [
    (df['Forest_Change_%'] < -0.1),
    (df['Forest_Change_%'] > 0.1)
]
wartosci = [2, 0]
df['Risk_Class'] = np.select(warunki, wartosci, default=1)

print("Tworzenie opóźnień czasowych (Lags)...")
# 4. Przesuwamy dane o rolnictwie i bydle o 1 rok.
# Dzięki temu model zobaczy w jednym wierszu: Las z 2015 i Rolnictwo z 2014.
df['Agri_Area_Lag1'] = df.groupby('Area')['Agri_Area'].shift(1)
df['Cattle_Head_Lag1'] = df.groupby('Area')['Cattle_Head'].shift(1)

# 1. Dynamika stada - czy bydła przybywa (rok do roku)?
df['Cattle_Change_%'] = df.groupby('Area')['Cattle_Head'].pct_change() * 100

# 2. Interakcja: Ile krów przypada na hektar rolniczy? (Gęstość wypasu)
df['Cattle_Density'] = df['Cattle_Head'] / (df['Agri_Area'] + 1) # +1 żeby nie dzielić przez zero

# 3. Opóźnienia dla nowych cech
df['Cattle_Change_Lag1'] = df.groupby('Area')['Cattle_Change_%'].shift(1)
df['Cattle_Density_Lag1'] = df.groupby('Area')['Cattle_Density'].shift(1)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)
# Ponowne czyszczenie i zapis
df = df.dropna()
df.to_csv('mvp_features_v2.csv', index=False)
print("Nowe cechy wygenerowane! Plik v2 gotowy.")

# Zapis do nowego pliku
df.to_csv('mvp_features.csv', index=False)
print(f"SUKCES! Baza pod model gotowa. Plik 'mvp_features.csv' ma {len(df)} rekordów do nauki.")