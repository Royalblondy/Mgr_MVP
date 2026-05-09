import wbgapi as wb
import pandas as pd
import time
import numpy as np

# Zestaw 11 potężnych wskaźników z Banku Światowego (Ekonomia, Geografia, Globalizacja, Rolnictwo)
indicators = {
    'AG.LND.FRST.ZS': 'Forest_Area_%',          # Zmienna docelowa / baza
    'AG.LND.ARBL.ZS': 'Arable_Land_%',          # Rolnictwo - pow. orna
    'NV.AGR.TOTL.ZS': 'Agriculture_GDP_%',      # Uzależnienie gospodarki od rolnictwa
    'NY.GDP.PCAP.CD': 'GDP_per_Capita',         # Bogactwo (Krzywa Kuznetsa)
    'EN.POP.DNST': 'Pop_Density',               # Demografia i presja na przestrzeń
    'SP.URB.GROW': 'Urban_Pop_Growth_%',        # Migracja do miast
    'EG.CFT.ACCS.RU.ZS': 'Rural_Clean_Fuel_%',  # Zapotrzebowanie na drewno opałowe
    'AG.LND.PRCP.MM': 'Average_Precipitation',  # Klimat (Opady - dżungla vs pustynia)
    'AG.LND.TOTL.K2': 'Land_Area_sqkm',         # Rozmiar państwa
    'TX.VAL.AGRI.ZS.UN': 'Agri_Exports_%',      # Eksport surowców rolniczych (Globalizacja)
    'DT.DOD.DECT.GN.ZS': 'External_Debt_%'      # Dług publiczny zmuszający do wycinki
}

print("1. Pobieranie danych z API Banku Światowego (Metoda pasterkowania)...")
ramki_danych = []

for code, name in indicators.items():
    print(f" -> Pobieranie: {name} ({code})...")
    try:
        df_ind = wb.data.DataFrame(code, time=range(2000, 2023), numericTimeKeys=True)
        df_ind = df_ind.reset_index()
        df_ind.columns = [str(c) for c in df_ind.columns]
        df_long = df_ind.melt(id_vars=['economy'], var_name='Year', value_name=name)
        df_long['Year'] = df_long['Year'].astype(int)
        ramki_danych.append(df_long)
        time.sleep(1.5) # Chłodzenie serwerów, żeby uniknąć Bad Gateway
    except Exception as e:
        print(f" ❌ Błąd przy pobieraniu {name}: {e}")

print("\n2. Łączenie wszystkich wskaźników...")
df_wdi = ramki_danych[0]
for i in range(1, len(ramki_danych)):
    df_wdi = pd.merge(df_wdi, ramki_danych[i], on=['economy', 'Year'], how='outer')

print("3. Wyciąganie metadanych Geograficznych (Absolutna Szerokość Geograficzna)...")
# API Banku Światowego ma ukryte dane o współrzędnych krajów!
kraje_meta = wb.economy.DataFrame().reset_index()

# Interesuje nas tylko nazwa i współrzędne
kraje_geo = kraje_meta[['id', 'name', 'latitude']].copy()

# Obliczamy 'Absolute_Latitude' (Odległość od równika w stopniach)
# Równik = 0 (Lasy deszczowe), Bieguny = 90 (Tajga/Tundra). Funkcja abs() usuwa minusy dla półkuli południowej.
kraje_geo['Absolute_Latitude'] = kraje_geo['latitude'].abs()

print("4. Czyszczenie bazy ze sztucznych agregatów (Świat, Europa itp.)...")
# Prawdziwe państwa mają współrzędne geograficzne, agregaty makro (np. "World") ich nie mają!
df_wdi_geo = pd.merge(df_wdi, kraje_geo, left_on='economy', right_on='id', how='left')
df_final = df_wdi_geo.dropna(subset=['latitude']).copy()

# Kosmetyka kolumn
df_final = df_final.rename(columns={'name': 'Country_Name'})
df_final = df_final.drop(columns=['economy', 'id', 'latitude'])

# Przesuwamy Country_Name i Year na sam początek tabeli
cols = ['Country_Name', 'Year'] + [c for c in df_final.columns if c not in ['Country_Name', 'Year']]
df_final = df_final[cols]

df_final = df_final.sort_values(by=['Country_Name', 'Year']).reset_index(drop=True)

print("\n5. Zapisywanie Master Bazy...")
df_final.to_csv('1_nowa_master_baza_wdi.csv', index=False)
print("✅ Gotowe! Mamy nowy, potężny fundament w pliku '1_nowa_master_baza_wdi.csv'.")