import pandas as pd
import matplotlib.pyplot as plt

print("Filtrowanie danych dla Europy...")
df = pd.read_csv('mvp_features.csv')

# Lista kilku krajów europejskich do porównania
kraje_ue = ['Poland', 'Germany', 'France', 'Italy', 'Spain', 'Sweden']
df_ue = df[df['Area'].isin(kraje_ue)].copy()

# 1. WYKRES: Trend zalesienia w Polsce
print("Generuję trend dla Polski...")
df_pl = df[df['Area'] == 'Poland'].copy()

plt.figure(figsize=(10, 6))
plt.plot(df_pl['Year'], df_pl['Forest_Area'], color='darkgreen', linewidth=3, marker='o', label='Las w PL')
plt.title('Polska: Powierzchnia lasów na przestrzeni lat')
plt.xlabel('Rok')
plt.ylabel('Hektary (ha)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('wykres_6_polska_las.png')
plt.close()

# 2. WYKRES PORÓWNAWCZY: Zmiana % lasów (Europa vs Reszta Świata)
print("Generuję porównanie kontynentalne...")
# Tutaj użyjemy triku: FAO często ma w kolumnie 'Area' gotowe regiony jak 'Europe'
# Jeśli nie, możemy po prostu pogrupować nasze kraje UE
df_eu_avg = df_ue.groupby('Year')['Forest_Change_%'].mean()

plt.figure(figsize=(10, 6))
plt.bar(kraje_ue, df_ue.groupby('Area')['Forest_Change_%'].mean(), color='skyblue')
plt.axhline(0, color='red', linewidth=1)
plt.title('Średnia roczna zmiana zalesienia w wybranych krajach Europy (%)')
plt.ylabel('Zmiana (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('wykres_7_europa_porownanie.png')
plt.close()

print("Gotowe! Sprawdź pliki 'wykres_6_polska_las.png' (trend w PL) oraz 'wykres_7_europa_porownanie.png'.")