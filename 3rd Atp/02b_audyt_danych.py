import pandas as pd
import numpy as np

print("1. Ładowanie finalnej bazy do audytu...")
df = pd.read_csv('3_finalna_baza_do_modelowania.csv')

print("\n" + "="*50)
print(" 📊 RAPORT GŁÓWNEGO INŻYNIERA DANYCH")
print("="*50)

print(f"\nLiczba wierszy (obserwacji): {len(df)}")
print(f"Liczba unikalnych krajów: {df['Country_Name'].nunique()}")

print("\n--- 1. KOMPLETNOŚĆ DANYCH (Ile procent to braki?) ---")
# Obliczamy procent brakujących danych w każdej kolumnie
braki = (df.isna().sum() / len(df)) * 100
braki = braki[braki > 0].sort_values(ascending=False)
if braki.empty:
    print("Braków Danych: 0% ! Perfekcyjna macierz.")
else:
    print(braki.round(2).astype(str) + ' %')

print("\n--- 2. EKSTREMALNE SKOKI (Analiza Delt) ---")
# Szukamy anomalii w dynamice zmian
kolumny_delt = [col for col in df.columns if 'Delta' in col]
stats_delt = df[kolumny_delt].describe().T[['min', 'max', 'mean']]

print("Sprawdzamy, czy jakaś zmienna nie urosła o absurdalne wartości:")
# Formatujemy do procentów dla lepszej czytelności (np. 1.0 to 100%)
stats_delt_proc = (stats_delt * 100).round(2).astype(str) + ' %'
print(stats_delt_proc.to_string())

print("\n--- 3. BALANS TARGETU (Zmiana Lasu) ---")
target = df['Forest_Area_%_Delta']
rosnie = len(target[target > 0])
spada = len(target[target < 0])
stoi = len(target[target == 0])

print(f"Lasy rosną (Zalesianie):   {rosnie} przypadków ({(rosnie/len(df))*100:.1f}%)")
print(f"Lasy znikają (Wylesianie): {spada} przypadków ({(spada/len(df))*100:.1f}%)")
print(f"Brak zmian:                {stoi} przypadków ({(stoi/len(df))*100:.1f}%)")
print("="*50)