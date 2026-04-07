import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Ładowanie danych
df = pd.read_csv('mvp_features_v2.csv')

# Odtwarzamy cechy w locie
df['Forest_Area_Lag1'] = df.groupby('Area')['Forest_Area'].shift(1)
df = df.dropna()
df['Area'] = df['Area'].astype('category')

# 2. PODZIAŁ CZASOWY
train_df = df[df['Year'] <= 2015].copy()
val_df   = df[(df['Year'] > 2015) & (df['Year'] <= 2019)].copy()
test_df  = df[df['Year'] >= 2020].copy()

# 3. ZMIANA TARGETU NA CIĄGŁY (REGRESJA)
features = [
    'Area',
    'Forest_Area_Lag1',
    'Agri_Area_Lag1',
    'Cattle_Head_Lag1',
    'Cattle_Change_Lag1',
    'Cattle_Density_Lag1'
]
target = 'Forest_Change_%' # Zgadujemy dokładny procent zmiany!

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Trenowanie modelu XGBRegressor...")

# 4. XGBOOST W WERSJI REGRESYJNEJ
model = xgb.XGBRegressor(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)

model.fit(X_train, y_train)

# 5. PRZEWIDYWANIE
y_pred = model.predict(X_test)

# 6. WYLICZANIE METRYK BŁĘDU
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- METRYKI REGRESJI (Dla testowych lat 2020+) ---")
print(f"MAE (Średni błąd bezwzględny): {mae:.4f}")
print(f"RMSE (Pierwiastek błędu średniokwadratowego): {rmse:.4f}")
print(f"R^2 (Współczynnik determinacji): {r2:.4f}")