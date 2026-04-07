import pandas as pd

# --- NAZWY PLIKÓW ---
PLIK_CROPS = 'Crops.csv'
PLIK_LASY = 'ForestAll.csv'
# --------------------------------

print("Rozpoczynam przygotowanie danych...")


def przetworz_szeroki_fao(nazwa_pliku):
    print(f" Wczytywanie pliku: {nazwa_pliku}...")
    # Czytamy pliki ze średnikami
    df = pd.read_csv(nazwa_pliku, sep=';', encoding='utf-8', low_memory=False)

    # Bierzemy tylko Area, Item, Element i kolumny lat (Y1961 itd.)
    kolumny_lat = [col for col in df.columns if col.startswith('Y') and not col.endswith(('F', 'N'))]
    df_filtered = df[['Area', 'Item', 'Element'] + kolumny_lat].copy()

    # "Topimy" tabelę (format szeroki -> długi)
    df_melted = df_filtered.melt(id_vars=['Area', 'Item', 'Element'], var_name='Year', value_name='Value')

    # Czyścimy z "Y" i konwertujemy na liczby
    df_melted['Year'] = df_melted['Year'].str.replace('Y', '')
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    # Wywalamy puste wiersze
    return df_melted.dropna(subset=['Year', 'Value'])


# 1. Przetwarzamy pierwszy plik
df_crops_livestock = przetworz_szeroki_fao(PLIK_CROPS)

#  - Wyciągamy Powierzchnię Rolniczą
df_rolnictwo = df_crops_livestock[df_crops_livestock['Element'] == 'Area harvested'].groupby(['Area', 'Year'])[
    'Value'].sum().reset_index()
df_rolnictwo.rename(columns={'Value': 'Agri_Area'}, inplace=True)

#  - Wyciągamy Bydło
df_bydlo = df_crops_livestock[df_crops_livestock['Item'].str.contains('Cattle', case=False, na=False)].groupby(['Area', 'Year'])[
    'Value'].sum().reset_index()
df_bydlo.rename(columns={'Value': 'Cattle_Head'}, inplace=True)

# 2. Przetwarzamy drugi plik
df_lasy_raw = przetworz_szeroki_fao(PLIK_LASY)
df_lasy = df_lasy_raw.groupby(['Area', 'Year'])['Value'].sum().reset_index()
df_lasy.rename(columns={'Value': 'Forest_Area'}, inplace=True)

# 3. Łączenie w ostateczny plik MVP
print(" Łączenie tabel...")
df_merged = pd.merge(df_lasy, df_rolnictwo, on=['Area', 'Year'], how='inner')
df_final = pd.merge(df_merged, df_bydlo, on=['Area', 'Year'], how='inner')

# Zapis do pliku
df_final.to_csv('mvp_data_global.csv', index=False)
print(f" SUKCES! Gotowy plik 'mvp_data_global.csv' zawiera {len(df_final)} wierszy.")