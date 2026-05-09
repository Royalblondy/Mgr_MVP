import pandas as pd

print("1. Ładowanie złączonej bazy...")
df = pd.read_csv('3_super_baza_modelowa.csv')

# Upewniamy się, że dane są posortowane po kraju i po roku! (Kluczowe dla przesunięć czasowych)
df = df.sort_values(by=['Country_Name', 'Year']).reset_index(drop=True)

print("2. Obliczanie Zmiennej Docelowej (Targetu)...")
# Zmienna docelowa: Zmiana powierzchni lasu rok do roku (w punktach procentowych)
df['Forest_Change_%'] = df.groupby('Country_Name')['Forest_Area_%'].diff()

print("3. Generowanie opóźnień czasowych (Lags)...")
# Tworzymy listę cech, które model powinien widzieć "z perspektywy poprzedniego roku"
cechy_do_opoznienia = [
    'Forest_Area_%', 'Arable_Land_%', 'Pop_Density', 'Rural_Elec_Access_%',
    'Rural_Clean_Fuel_%', 'Renewable_Energy_%', 'Urban_Pop_Growth_%',
    'Population_Total', 'Export quantity', 'Import quantity', 'Production'
]

# Dla każdej cechy tworzymy nową kolumnę z wartością z poprzedniego roku
for cecha in cechy_do_opoznienia:
    nowa_nazwa = cecha.replace('%', '').replace(' ', '_').strip('_') + '_Lag1'
    df[nowa_nazwa] = df.groupby('Country_Name')[cecha].shift(1)

print("4. Sprzątanie po inżynierii czasu...")
# Zauważ: Rok 2000 nie ma swojego "poprzedniego roku" (1999), więc wygenerował wartości NaN (puste).
# Musimy odciąć ten pierwszy, "niekompletny" rok dla każdego kraju.
df_clean = df.dropna(subset=['Forest_Change_%']).copy()

# Wyrzucamy stare, nieprzesunięte kolumny (żeby model "nie oszukiwał" patrząc w przyszłość)
kolumny_do_zachowania = ['Country_Name', 'Year', 'Forest_Change_%'] + [
    col for col in df_clean.columns if '_Lag1' in col
]
df_final = df_clean[kolumny_do_zachowania]

print("5. Zapisywanie ostatecznej macierzy do Machine Learningu...")
df_final.to_csv('4_baza_gotowa_na_ml.csv', index=False)
print("Gotowe! Plik '4_baza_gotowa_na_ml.csv' czeka. Wchodzimy w Fazę Algorytmów!")