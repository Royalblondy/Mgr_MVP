import pandas as pd

print("1. Ładowanie danych z Banku Światowego i FAO...")
df_wb = pd.read_csv('1_dane_bank_swiatowy_czyste.csv')
df_fao = pd.read_csv('2_dane_fao.csv')

print("2. Czyszczenie Banku Światowego z agregatów i makroregionów...")
# Tworzymy listę słów-kluczy, które zdradzają, że dany wiersz to nie jest prawdziwe państwo
slowa_zakazane = [
    'World', 'income', 'IDA', 'IBRD', 'dividend', 'Sub-Saharan',
    'Europe &', 'Latin America', 'East Asia', 'South Asia',
    'Middle East', 'European Union', 'Euro area', 'OECD',
    'Heavily indebted', 'Small states', 'Central Europe'
]

# Filtracja: zostawiamy tylko te wiersze, gdzie nazwa kraju NIE zawiera zakazanych słów
maska_prawdziwe_kraje = ~df_wb['Country_Name'].str.contains('|'.join(slowa_zakazane), case=False, na=False)
df_wb = df_wb[maska_prawdziwe_kraje]

print("3. Formatowanie danych leśnych (FAO)...")
fao_pivot = df_fao.pivot_table(index=['Area', 'Year'], columns='Element', values='Value').reset_index()

fao_pivot = fao_pivot.rename(columns={
    'Area': 'Country_Name',
    'Production Quantity': 'Roundwood_Production_m3',
    'Import Quantity': 'Roundwood_Import_m3',
    'Export Quantity': 'Roundwood_Export_m3'
})

# --- Słownik tłumaczący z FAO na Bank Światowy (Uratowani gracze!) ---
mapa_krajow = {
    'United States of America': 'United States',
    'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
    'Congo': 'Congo, Rep.',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Viet Nam': 'Vietnam',
    'United Republic of Tanzania': 'Tanzania',
    'Venezuela (Bolivarian Republic of)': 'Venezuela, RB',
    'Iran (Islamic Republic of)': 'Iran, Islamic Rep.',
    'Egypt': 'Egypt, Arab Rep.',
    'Republic of Korea': 'Korea, Rep.',
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    'Slovakia': 'Slovak Republic',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Yemen': 'Yemen, Rep.',
    'Gambia': 'Gambia, The',
    'Bahamas': 'Bahamas, The',
    'Türkiye': 'Turkiye',
    'Russian Federation': 'Russian Federation'
}

fao_pivot['Country_Name'] = fao_pivot['Country_Name'].replace(mapa_krajow)

print("4. Wielka Fuzja (Inner Join)...")
df_merged = pd.merge(df_wb, fao_pivot, on=['Country_Name', 'Year'], how='inner')

kraje_wb = set(df_wb['Country_Name'].unique())
kraje_po_fuzji = set(df_merged['Country_Name'].unique())
utracone_kraje = kraje_wb - kraje_po_fuzji

print(f"\n STATYSTYKI FUZJI:")
print(f"Kraje po wyczyszczeniu śmieci z Banku Światowego: {len(kraje_wb)}")
print(f"Kraje pomyślnie połączone z FAO: {len(kraje_po_fuzji)}")
print(f"Utracono 'krajów' (głównie nieznaczące małe wyspy): {len(utracone_kraje)}")

print("\n5. Zapisywanie ostatecznej bazy...")
df_merged.to_csv('3_super_baza_modelowa.csv', index=False)
print("Gotowe! Plik '3_super_baza_modelowa.csv' czeka na dysku.")