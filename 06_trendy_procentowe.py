import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mvp_features.csv')


def wykres_tempa_zmian(kraj, kolor):
    dane_kraju = df[df['Area'] == kraj]

    plt.figure(figsize=(12, 5))
    # Rysujemy linię procentowej zmiany
    plt.plot(dane_kraju['Year'], dane_kraju['Forest_Change_%'],
             color=kolor, linewidth=2, marker='.', label=f'Zmiana roczna % ({kraj})')

    # Dodajemy linię zero - wszystko poniżej to strata, powyżej to zysk
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)

    # Zaznaczamy strefę ryzyka (poniżej -0.1%)
    plt.axhline(-0.1, color='red', linestyle='--', alpha=0.5, label='Próg Ryzyka (Spadek)')

    plt.title(f'Dynamika zmian lasów: {kraj} (rok do roku)')
    plt.ylabel('Zmiana powierzchni (%)')
    plt.xlabel('Rok')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'wykres_dynamika_{kraj.lower()}.png')
    plt.close()


# Generujemy dla dwóch różnych światów
wykres_tempa_zmian('Sweden', 'blue')
wykres_tempa_zmian('Brazil', 'orange')
wykres_tempa_zmian('Poland', 'green')

print("Gotowe!")