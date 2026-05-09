import pandas as pd

print("Ładowanie złączonej super-bazy danych...")
df = pd.read_csv('3_super_baza_modelowa.csv')

# Kolumny, które są naszymi cechami (wszystko poza krajem i rokiem)
cechy = [col for col in df.columns if col not in ['economy', 'Country_Name', 'Year']]

print("\n" + "=" * 60)
print("  OSTATECZNY RAPORT KOMPLETNOŚCI DANYCH (Próg 80%)")
print("=" * 60)

najpozniejszy_rok = 2000

for cecha in cechy:
    # Liczymy, jaki % państw ma wpisaną wartość w danym roku
    kompletnosc_po_roku = df.groupby('Year')[cecha].apply(lambda x: x.notnull().mean() * 100)

    # Szukamy lat, gdzie mamy min. 80% danych
    lata_ok = kompletnosc_po_roku[kompletnosc_po_roku >= 80].index.tolist()

    if lata_ok:
        start_rok = min(lata_ok)
        print(f"ok {cecha:<30}: Kompletne w >80% od {start_rok} r.")
        if start_rok > najpozniejszy_rok:
            najpozniejszy_rok = start_rok
    else:
        max_kompletnosc = kompletnosc_po_roku.max()
        rok_max = kompletnosc_po_roku.idxmax()
        print(f"not ok {cecha:<30}: NIGDY nie osiąga 80%! (Max: {max_kompletnosc:.1f}% w {rok_max})")

print("-" * 60)
print(f"💡 WNIOSEK OSTATECZNY: Najbezpieczniej trenować nasz model na latach od: {najpozniejszy_rok}")
print("-" * 60)

print("\n" + "=" * 60)
print(" 🕵️‍♂️ AUDYT ANOMALII (Sprawdzenie logiki nowych danych)")
print("=" * 60)

for cecha in cechy:
    minimum = df[cecha].min()
    maksimum = df[cecha].max()
    srednia = df[cecha].mean()

    print(f"Zmienna: {cecha}")
    if maksimum > 1000000:  # Jeśli wartości są w milionach (jak np. populacja lub drewno)
        print(f"  Min: {minimum:,.0f} | Średnia: {srednia:,.0f} | Max: {maksimum:,.0f}")
    else:
        print(f"  Min: {minimum:.2f} | Średnia: {srednia:.2f} | Max: {maksimum:.2f}")
    print("-" * 40)