import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Ładowanie danych
df = pd.read_csv('mvp_features_v2.csv')

# --- CHIRURGICZNE CIĘCIE: Usuwamy agregaty FAO i mikropaństwa ---
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
    # Mikropaństwa i anomalie środowiskowe
    'Bahrain', 'Malta', 'Iceland'
]

# Odfiltrowujemy wszystko, co jest na czarnej liście
df_clean = df[~df['Area'].isin(czarna_lista)].copy()

# Przygotowanie cech z opóźnieniem
df_clean['Forest_Area_Lag1'] = df_clean.groupby('Area')['Forest_Area'].shift(1)
df_clean = df_clean.dropna()
df_clean['Area'] = df_clean['Area'].astype('category')

# 2. PODZIAŁ CZASOWY
train_df = df_clean[df_clean['Year'] <= 2015].copy()
test_df  = df_clean[df_clean['Year'] >= 2020].copy()

features = ['Area', 'Forest_Area_Lag1', 'Agri_Area_Lag1', 'Cattle_Head_Lag1', 'Cattle_Change_Lag1', 'Cattle_Density_Lag1']
target = 'Forest_Change_%'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 3. TRENOWANIE CZYSTEGO MODELU
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)

model.fit(X_train, y_train)

# 4. PRZEWIDYWANIE I METRYKI
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- OSTATECZNE METRYKI (Po usunięciu agregatów) ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
print(f"Liczba wierszy w teście: {len(X_test)}")