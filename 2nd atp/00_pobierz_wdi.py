import wbgapi as wb
import pandas as pd
import time

indicators = {
    'AG.LND.FRST.ZS': 'Forest_Area_%',
    'AG.LND.ARBL.ZS': 'Arable_Land_%',
    'EN.POP.DNST': 'Pop_Density',
    'EG.ELC.ACCS.RU.ZS': 'Rural_Elec_Access_%',
    'EG.CFT.ACCS.RU.ZS': 'Rural_Clean_Fuel_%',
    'EG.FEC.RNEW.ZS': 'Renewable_Energy_%',
    'SP.URB.GROW': 'Urban_Pop_Growth_%',
    'SP.POP.TOTL': 'Population_Total'
}

print("1. Łączenie z API Banku Światowego...")
print("Pobieramy wskaźniki pojedynczo, aby nie przeciążyć serwera WDI.")

ramki_danych = []

# Pobieramy każdy wskaźnik osobną paczką
for code, name in indicators.items():
    print(f" -> Pobieranie: {name}...")
    try:
        # Pobranie pojedynczego wskaźnika
        df_ind = wb.data.DataFrame(code, time=range(2000, 2023), numericTimeKeys=True)
        df_ind = df_ind.reset_index()

        # Melt od razu dla tego wskaźnika
        df_ind.columns = [str(c) for c in df_ind.columns]
        df_long = df_ind.melt(id_vars=['economy'], var_name='Year', value_name=name)

        ramki_danych.append(df_long)
        time.sleep(1)  # Sekunda oddechu dla serwera Banku Światowego
    except Exception as e:
        print(f" Błąd przy pobieraniu {name}: {e}")

print("\n2. Formatowanie danych i łączenie pobranych paczek...")
# Sklejanie wszystkich 7 tabel w jedną po kraju i roku
df_final = ramki_danych[0]
for i in range(1, len(ramki_danych)):
    df_final = pd.merge(df_final, ramki_danych[i], on=['economy', 'Year'], how='outer')

print("3. Kosmetyka (Pełne nazwy krajów)...")
countries = wb.economy.DataFrame()['name']
df_final.insert(1, 'Country_Name', df_final['economy'].map(countries))

df_final = df_final.sort_values(by=['Country_Name', 'Year']).reset_index(drop=True)

print("4. Zapis do pliku...")
df_final.to_csv('1_dane_bank_swiatowy_czyste.csv', index=False)
print("Gotowe! Mamy czystą baze w pliku '1_dane_bank_swiatowy_czyste.csv'.")