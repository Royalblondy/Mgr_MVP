import pandas as pd
import numpy as np

print("--- Ładowanie bazy do analizy statystycznej ---")
df = pd.read_csv('4_baza_gotowa_na_ml.csv')

print(f"\n1. ROZMIAR ZBIORU TRENINGOWEGO:")
print(f"Liczba wierszy (obserwacji Kraj-Rok): {df.shape[0]}")
print(f"Liczba kolumn (cechy + target): {df.shape[1]}")

print("\n2. ANALIZA ZMIENNEJ DOCELOWEJ (Forest_Change_%):")
target = df['Forest_Change_%']
print(f"Średnia roczna zmiana: {target.mean():.4f} p.p.")
print(f"Mediana zmiany: {target.median():.4f} p.p.")
print(f"Największy roczny spadek: {target.min():.4f} p.p.")
print(f"Największy roczny wzrost: {target.max():.4f} p.p.")

wzrosty = len(df[df['Forest_Change_%'] > 0])
spadki = len(df[df['Forest_Change_%'] < 0])
bez_zmian = len(df[df['Forest_Change_%'] == 0])
total = len(df)

print(f"\nRozkład trendów leśnych w bazie (od 2000 roku):")
print(f" Wzrosty zalesienia: {wzrosty} ({wzrosty/total*100:.1f}%)")
print(f" Spadki zalesienia:  {spadki} ({spadki/total*100:.1f}%)")
print(f" Brak zmian:         {bez_zmian} ({bez_zmian/total*100:.1f}%)")

print("\n3. TOP 5 NAJSILNIEJSZYCH KORELACJI Z TARGETEM (Pearson):")
# Wykluczamy kolumny tekstowe do korelacji
cols_num = df.select_dtypes(include=[np.number]).columns
korelacje = df[cols_num].corr()['Forest_Change_%'].sort_values(key=abs, ascending=False)
# Pomijamy sam target (indeks 0) i wyświetlamy 5 kolejnych
print(korelacje[1:6].to_string())

print("\n4. PODSTAWOWE STATYSTYKI PREDYKTORÓW (Cechy Lag1):")
features = [col for col in df.columns if '_Lag1' in col]
stats = df[features].describe().T[['mean', 'min', 'max', 'std']]
# Formatowanie wyświetlania dużych liczb i procentów
pd.options.display.float_format = '{:,.2f}'.format
print(stats)