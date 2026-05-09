import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

print("Ładowanie i czyszczenie danych...")
df = pd.read_csv('mvp_features_v2.csv')

czarna_lista = [
    'World', 'Africa', 'Eastern Africa', 'Middle Africa', 'Northern Africa',
    'Southern Africa', 'Western Africa', 'Americas', 'Northern America',
    'Central America', 'Caribbean', 'South America', 'Asia', 'Central Asia',
    'Eastern Asia', 'Southern Asia', 'South-eastern Asia', 'Western Asia',
    'Europe', 'Eastern Europe', 'Northern Europe', 'Southern Europe',
    'Western Europe', 'Oceania', 'Australia and New Zealand', 'Melanesia',
    'Micronesia', 'Polynesia', 'European Union (27)', 'European Union (28)',
    'Least Developed Countries', 'Land Locked Developing Countries (LLDCs)',
    'Land Locked Developing Countries', 'Small Island Developing States',
    'Low Income Food Deficit Countries', 'Net Food Importing Developing Countries',
    'Annex I countries', 'Non-Annex I countries', 'OECD',
    'Bahrain', 'Malta', 'Iceland'
]

df_clean = df[~df['Area'].isin(czarna_lista)].copy()
df_clean['Forest_Area_Lag1'] = df_clean.groupby('Area')['Forest_Area'].shift(1)
df_clean = df_clean.dropna()
df_clean['Area'] = df_clean['Area'].astype('category')

train_df = df_clean[df_clean['Year'] <= 2015].copy()
test_df = df_clean[df_clean['Year'] >= 2020].copy()

features = ['Area', 'Forest_Area_Lag1', 'Agri_Area_Lag1', 'Cattle_Head_Lag1', 'Cattle_Change_Lag1',
            'Cattle_Density_Lag1']
target = 'Forest_Change_%'

print("Trenowanie modelu...")
model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42, enable_categorical=True
)
model.fit(train_df[features], train_df[target])

# Doklejamy przewidywania do naszego zbioru testowego!
test_df['Predicted_Change_%'] = model.predict(test_df[features])

print("Generowanie wykresów...")
kraje_do_analizy = ['Poland', 'Brazil', 'Indonesia']

plt.figure(figsize=(15, 5))

for i, kraj in enumerate(kraje_do_analizy, 1):
    plt.subplot(1, 3, i)
    dane_kraju = test_df[test_df['Area'] == kraj].sort_values('Year')

    # Linia prawdy (Rzeczywistość FAO)
    plt.plot(dane_kraju['Year'], dane_kraju['Forest_Change_%'],
             color='red', marker='o', linewidth=2, label='Prawda (FAO)')

    # Linia modelu (Nasz XGBoost)
    plt.plot(dane_kraju['Year'], dane_kraju['Predicted_Change_%'],
             color='blue', marker='x', linestyle='--', linewidth=2, label='Model (XGBoost)')

    plt.title(f'{kraj}: Prawda vs Model')
    plt.xlabel('Rok (Test: 2020+)')
    plt.ylabel('Zmiana zalesienia (%)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
    plt.xticks(dane_kraju['Year'].astype(int))  # Wymuszamy liczby całkowite na osi X
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wykres_9_prawda_vs_model.png')
plt.close()

print("Gotowe! Otwórz plik 'wykres_9_prawda_vs_model.png'.")