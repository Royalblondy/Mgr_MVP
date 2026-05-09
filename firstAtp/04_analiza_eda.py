import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Ładowanie danych do pogłębionej analizy...")
df = pd.read_csv('mvp_features.csv')

# 1. MACIERZ KORELACJI (Heatmapa)
print("Generuję macierz korelacji...")
plt.figure(figsize=(10, 8))
# Wybieramy tylko kolumny liczbowe do korelacji
cols_to_corr = ['Forest_Area', 'Agri_Area', 'Cattle_Head', 'Forest_Change_%', 'Agri_Area_Lag1', 'Cattle_Head_Lag1']
correlation_matrix = df[cols_to_corr].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('Korelacja między zmiennymi (Cały Świat)')
plt.tight_layout()
plt.savefig('wykres_3_korelacje.png')
plt.close()

# 2. WYKRES PUNKTOWY: Bydło vs Zmiana Lasów
print("Generuję wykres punktowy: Bydło vs Zmiana Lasów...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Cattle_Head_Lag1', y='Forest_Change_%', alpha=0.5, hue='Risk_Class', palette='RdYlGn')
plt.axhline(0, color='black', linestyle='--') # Linia zero
plt.xscale('log') # Skala logarytmiczna, bo kraje mają bardzo różną liczbę krów
plt.title('Wpływ liczby bydła (rok wcześniej) na zmianę lasów')
plt.xlabel('Liczba bydła (skala logarytmiczna)')
plt.ylabel('Zmiana powierzchni lasów (%)')
plt.tight_layout()
plt.savefig('wykres_4_bydlo_vs_las.png')
plt.close()

# 3. TOP 10 KRAJÓW Z NAJWIĘKSZYM SPADKIEM LASÓW (średniorocznie)
print("Generuję ranking krajów...")
top_deforest = df.groupby('Area')['Forest_Change_%'].mean().sort_values().head(10)
plt.figure(figsize=(10, 6))
top_deforest.plot(kind='barh', color='red')
plt.title('Top 10 krajów z największym średnim spadkiem zalesienia (%)')
plt.xlabel('Średnia zmiana roczna (%)')
plt.tight_layout()
plt.savefig('wykres_5_ranking_spadku.png')
plt.close()

print("Gotowe! 3 nowe wykresy w folderze: wykres_3, wykres_4 i wykres_5.")


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