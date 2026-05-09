import pandas as pd
import matplotlib.pyplot as plt

print("Wczytuję gotowe dane MVP...")
df = pd.read_csv('mvp_features.csv')

# --- WYKRES 1: Rozkład klas ryzyka na świecie ---
print("Generuję Wykres 1 (Klasy Ryzyka)...")
plt.figure(figsize=(8, 6))
# Zliczamy ile jest poszczególnych klas
counts = df['Risk_Class'].value_counts().sort_index()

# Podmiana numerków na nazwy dla czytelności
labels = ['0 - Wzrost', '1 - Stabilnie', '2 - Spadek (Ryzyko)']
colors = ['green', 'gray', 'red']

bars = plt.bar(labels, counts.values, color=colors)
plt.title('Rozkład klas ryzyka zalesienia (Cały Świat)')
plt.ylabel('Liczba obserwacji (Kraj-Rok)')

# Dodanie wartości nad słupkami
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 100, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('wykres_1_ryzyko.png')
plt.close()

# --- WYKRES 2: Trend dla wybranego kraju (Brazylia) ---
print("Generuję Wykres 2 (Brazylia - Las vs Rolnictwo)...")
df_brazil = df[df['Area'] == 'Brazil'].copy()

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:green'
ax1.set_xlabel('Rok')
ax1.set_ylabel('Powierzchnia Lasów (ha)', color=color)
ax1.plot(df_brazil['Year'], df_brazil['Forest_Area'], color=color, linewidth=2, label='Lasy')
ax1.tick_params(axis='y', labelcolor=color)

# Druga oś Y dla Rolnictwa (bo wartości są w innej skali)
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Powierzchnia Rolnicza (ha)', color=color)
ax2.plot(df_brazil['Year'], df_brazil['Agri_Area'], color=color, linewidth=2, linestyle='--', label='Rolnictwo')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Brazylia: Spadek zalesienia a wzrost powierzchni rolniczej')
fig.tight_layout()
plt.savefig('wykres_2_brazylia.png')
plt.close()

print("Gotowe! W folderze projektu pojawiły się dwa pliki: 'wykres_1_ryzyko.png' oraz 'wykres_2_brazylia.png'.")