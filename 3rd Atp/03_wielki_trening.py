import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1. Ładowanie ostatecznej bazy danych...")
df = pd.read_csv('3_finalna_baza_do_modelowania.csv')

print("2. Tarcza Inżynierska: Winsoryzacja (Usuwanie absurdalnych %)...")
kolumny_delt = [col for col in df.columns if 'Delta' in col]

for col in kolumny_delt:
    # Obliczamy 1. i 99. percentyl, by odciąć matematyczne anomalie
    dolna_granica = df[col].quantile(0.01)
    gorna_granica = df[col].quantile(0.99)
    # Wszystko poniżej 1% i powyżej 99% zostaje spłaszczone do tych granic
    df[col] = df[col].clip(lower=dolna_granica, upper=gorna_granica)

print("3. Inżynieria Czasu: Ustawianie Targetu na PRZYSZŁOŚĆ...")
df['Target_Forest_Delta_Next_Year'] = df.groupby('Country_Name')['Forest_Area_%_Delta'].shift(-1)
df = df.dropna(subset=['Target_Forest_Delta_Next_Year'])

print("4. Definiowanie zbioru uczącego i testowego...")
train_mask = df['Year'] <= 2018
test_mask = df['Year'] > 2018

kolumny_do_odrzucenia = [
    'Country_Name', 'Year', 'Target_Forest_Delta_Next_Year',
    'Forest_Area_%_Delta'
]
cechy_wejsciowe = [col for col in df.columns if col not in kolumny_do_odrzucenia]

X_train = df.loc[train_mask, cechy_wejsciowe]
y_train = df.loc[train_mask, 'Target_Forest_Delta_Next_Year']

X_test = df.loc[test_mask, cechy_wejsciowe]
y_test = df.loc[test_mask, 'Target_Forest_Delta_Next_Year']

print("\n5. Budowa i Trening Modelu XGBoost (Zoptymalizowany dla Delt)...")
model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("\n6. Egzamin na latach > 2018...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print(" WYNIKI OSTATECZNEGO MODELU GLOBALNEGO")
print("="*60)
print(f"R² (Współczynnik determinacji): {r2:.4f}")
print(f"MAE (Średni błąd bezwzględny):   {mae:.5f}")
print(f"RMSE (Błąd średniokwadratowy):   {rmse:.5f}")
print("="*60)

print("\n7. Tabela Ważności Cech - Co napędza wylesianie?")
waznosc = pd.DataFrame({
    'Cecha': cechy_wejsciowe,
    'Waznosc_%': model.feature_importances_ * 100
}).sort_values(by='Waznosc_%', ascending=False)

print(waznosc.to_string(index=False))