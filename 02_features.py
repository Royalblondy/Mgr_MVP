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

# Czyszczenie - wyrzucamy puste wiersze powstałe na skutek przesunięć w czasie (brak danych dla pierwszego roku)
df = df.dropna()

# Zapis do nowego pliku
df.to_csv('mvp_features.csv', index=False)
print(f"SUKCES! Baza pod model gotowa. Plik 'mvp_features.csv' ma {len(df)} rekordów do nauki.")