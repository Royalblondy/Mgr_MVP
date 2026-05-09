import pandas as pd

print("Ładowanie pobranych danych...")
df = pd.read_csv('1_dane_bank_swiatowy_czyste.csv')

# Definiujemy, które kolumny to nasze wskaźniki (pomijamy nazwy krajów i rok)
cechy = [col for col in df.columns if col not in ['economy', 'Country_Name', 'Year']]

print("\n" + "=" * 50)
print("RAPORT KOMPLETNOŚCI DANYCH (Próg 80%)")
print("=" * 50)

najpozniejszy_rok = 1990

for cecha in cechy:
    # Dla każdego roku liczymy, jaki % państw ma wpisaną wartość (nie jest NaN)
    kompletnosc_po_roku = df.groupby('Year')[cecha].apply(lambda x: x.notnull().mean() * 100)

    # Filtrujemy tylko te lata, gdzie kompletność >= 80%
    lata_ok = kompletnosc_po_roku[kompletnosc_po_roku >= 80].index.tolist()

    if lata_ok:
        start_rok = min(lata_ok)
        print(f"ok {cecha:<25}: Kompletne w >80% od roku {start_rok}")
        if start_rok > najpozniejszy_rok:
            najpozniejszy_rok = start_rok
    else:
        max_kompletnosc = kompletnosc_po_roku.max()
        rok_max = kompletnosc_po_roku.idxmax()
        print(f"not ok {cecha:<25}: NIGDY nie osiąga 80%! (Max to {max_kompletnosc:.1f}% w {rok_max} r.)")

print("-" * 50)
print(f" WNIOSEK: Aby mieć >80% danych dla WSZYSTKICH powyższych cech, ")
print(f"należy obciąć zbiór danych i zacząć analizę od roku: {najpozniejszy_rok}")
print("-" * 50)

print("\n" + "=" * 50)
print(" SZYBKI AUDYT ANOMALII (Czy dane mają sens?)")
print("=" * 50)

for cecha in cechy:
    minimum = df[cecha].min()
    maksimum = df[cecha].max()
    srednia = df[cecha].mean()

    print(f"Zmienna: {cecha}")
    print(f"  Min: {minimum:.2f} | Średnia: {srednia:.2f} | Max: {maksimum:.2f}")

    # Proste testy logiczne
    if 'ZS' in cecha or '%' in cecha:  # Jeśli to wskaźnik procentowy
        if maksimum > 100.1:
            print(f" UWAGA: Wartość procentowa przekracza 100% ({maksimum:.2f})!")
        if minimum < 0:
            print(f" UWAGA: Wartość procentowa jest ujemna ({minimum:.2f})!")
    print("-" * 30)