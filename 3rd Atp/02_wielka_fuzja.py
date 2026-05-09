import pandas as pd
import numpy as np

print("1. Ładowanie danych...")
df_wdi = pd.read_csv('1_nowa_master_baza_wdi.csv')
df_roundwood = pd.read_csv('Roundwood.csv')
df_pastures = pd.read_csv('Permanent meadows and pastures.csv')

def clean_fao(df, value_name):
    # FAO używa 'Area' zamiast 'Country_Name'
    df = df.rename(columns={'Area': 'Country_Name', 'Value': value_name})
    return df[['Country_Name', 'Year', value_name]]

print("2. Czyszczenie i przygotowanie plików FAOSTAT...")
# W pliku Roundwood mamy zazwyczaj Produkcję i Eksport w jednej kolumnie 'Element'
# Musimy je rozdzielić
rw_prod = df_roundwood[df_roundwood['Element'] == 'Production'].copy()
rw_export = df_roundwood[df_roundwood['Element'].str.contains('Export', na=False)].copy()

rw_prod = clean_fao(rw_prod, 'Roundwood_Prod')
rw_export = clean_fao(rw_export, 'Roundwood_Export')
pastures = clean_fao(df_pastures, 'Pasture_Area')

print("3. Łączenie wszystkich źródeł...")
# Łączymy najpierw FAO
df_fao = pd.merge(rw_prod, rw_export, on=['Country_Name', 'Year'], how='outer')
df_fao = pd.merge(df_fao, pastures, on=['Country_Name', 'Year'], how='outer')

# Fuzja z Bankiem Światowym
df = pd.merge(df_wdi, df_fao, on=['Country_Name', 'Year'], how='inner')

print("4. Obliczanie wskaźników relatywnych (Pastures %)...")
# Zakładamy, że Pasture_Area jest w 1000 ha, a Land_Area_sqkm w km2 (1 km2 = 100 ha)
# Zatem: (Area_1000ha * 10) / Land_Area_km2 * 100 = %
df['Pasture_Land_%'] = (df['Pasture_Area'] * 10) / df['Land_Area_sqkm'] * 100

print("5. Inżynieria Delt (Dynamika zmian)...")
# Lista cech, dla których model musi widzieć "tempo", a nie tylko "stan"
cechy_do_delt = [
    'Forest_Area_%', 'Arable_Land_%', 'GDP_per_Capita',
    'Pop_Density', 'Pasture_Land_%', 'Roundwood_Prod'
]

df = df.sort_values(['Country_Name', 'Year'])

for cecha in cechy_do_delt:
    # Obliczamy różnicę rok do roku (np. wzrost o 2% to 0.02)
    df[f'{cecha}_Delta'] = df.groupby('Country_Name')[cecha].pct_change()

print("6. Finalne szlifowanie...")
df = df.replace([np.inf, -np.inf], 0).fillna(0)
# Usuwamy rok 2000 (brak delt)
df = df[df['Year'] > 2000]

df.to_csv('3_finalna_baza_do_modelowania.csv', index=False)
print(f"✅ Baza gotowa! Plik '3_finalna_baza_do_modelowania.csv' zawiera {len(df)} rekordów.")