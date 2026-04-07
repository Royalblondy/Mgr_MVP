import pandas as pd

# --- KONFIGURACJA ---
PLIK_LAND_USE = 'LandUse_Bulk.csv'
PLIK_CROPS = 'Crops.csv'


# --------------------

def wczytaj_bulk_land(nazwa_pliku):
    print(f"Przetwarzam wielki plik {nazwa_pliku}... to może chwilę potrwać.")
    # Bulk download zazwyczaj używa przecinków, ale sprawdzimy to w locie
    df = pd.read_csv(nazwa_pliku, encoding='latin1', low_memory=False)

    # Interesują nas tylko hektary (Area)
    df = df[df['Element'] == 'Area'].copy()

    # Wybieramy lata (kolumny Y1961, Y1962...)
    kolumny_lat = [col for col in df.columns if col.startswith('Y') and col[1:].isdigit()]
    df_filtered = df[['Area', 'Item'] + kolumny_lat].copy()

    # Topimy tabelę
    df_melted = df_filtered.melt(id_vars=['Area', 'Item'], var_name='Year', value_name='Value')
    df_melted['Year'] = df_melted['Year'].str.replace('Y', '').astype(int)
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    return df_melted.dropna(subset=['Value'])


# 1. Pobieramy Lasy i Rolnictwo z nowego pliku
df_land = wczytaj_bulk_land(PLIK_LAND_USE)

print("Wyodrębniam lasy i rolnictwo...")
df_lasy = df_land[df_land['Item'] == 'Forest land'].copy()
df_lasy = df_lasy.rename(columns={'Value': 'Forest_Area'})[['Area', 'Year', 'Forest_Area']]

df_rolnictwo = df_land[df_land['Item'] == 'Agricultural land'].copy()
df_rolnictwo = df_rolnictwo.rename(columns={'Value': 'Agri_Area'})[['Area', 'Year', 'Agri_Area']]

# 2. Pobieramy Bydło ze starego pliku (pamiętamy o średnikach)
print("Doczytuję bydło z pliku Crops...")
df_c = pd.read_csv(PLIK_CROPS, sep=';', encoding='utf-8', low_memory=False)
kol_lat_c = [col for col in df_c.columns if col.startswith('Y') and not col.endswith(('F', 'N'))]
df_c_filt = df_c[df_c['Item'].str.contains('Cattle', case=False, na=False)][['Area'] + kol_lat_c]
df_bydlo = df_c_filt.melt(id_vars=['Area'], var_name='Year', value_name='Cattle_Head')
df_bydlo['Year'] = df_bydlo['Year'].str.replace('Y', '').astype(int)
df_bydlo['Cattle_Head'] = pd.to_numeric(df_bydlo['Cattle_Head'], errors='coerce')
df_bydlo = df_bydlo.groupby(['Area', 'Year'])['Cattle_Head'].sum().reset_index()

# 3. Łączymy wszystko
print("Łączenie ostatecznej tabeli...")
m1 = pd.merge(df_lasy, df_rolnictwo, on=['Area', 'Year'], how='inner')
df_final = pd.merge(m1, df_bydlo, on=['Area', 'Year'], how='inner')

df_final.to_csv('mvp_data_global.csv', index=False)
print(f"SUKCES! Nowy, poprawny plik ma {len(df_final)} wierszy.")